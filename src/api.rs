use crate::state::AppState;
use actix_web::http::header::HeaderValue;
use actix_web::rt::time::interval;
use actix_web::web::Bytes;
use actix_web::{post, web, HttpResponse, Responder};
use actix_web_lab::sse;
use anyhow::Error;
use futures::StreamExt;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Serialize)]
pub struct TopResult {
    item: String,
    score: f32,
}

#[derive(Serialize)]
pub struct SimilarityResponse {
    top_results: Vec<TopResult>,
}

#[derive(Deserialize)]
pub struct SimilarityRequest {
    text: String,
    num_results: usize,
}

#[derive(Deserialize)]
pub struct GenerateTextRequest {
    prompt: String,
    max_length: usize,
}

#[derive(Serialize)]
pub struct GenerateTextResponse {
    generated_text: String,
}

// #[post("/find_similar")]
pub async fn find_similar(
    state: web::Data<AppState>,
    payload: web::Json<SimilarityRequest>,
) -> impl Responder {
    let bert_model = &state.bert_model;
    let text_map = &state.text_map;

    // Generate embedding for the input text
    let query_embedding = match bert_model.infer_sentence_embedding(&payload.text) {
        Ok(embedding) => embedding,
        Err(_) => return HttpResponse::InternalServerError().body("Failed to generate embedding"),
    };

    let results: Vec<(usize, f32)> = match bert_model
        .score_vector_similarity(query_embedding, payload.num_results)
    {
        Ok(res) => res,
        Err(_) => {
            return HttpResponse::InternalServerError().body("Failed to score vector similarity")
        }
    };

    let top_results: Vec<TopResult> = results
        .into_iter()
        .filter_map(|(idx, score)| {
            text_map.get(idx).map(|item| TopResult {
                item: item.to_string(),
                score,
            })
        })
        .collect();

    HttpResponse::Ok().json(SimilarityResponse { top_results })
}

// #[post("/generate_text")]
pub async fn generate_text(
    state: web::Data<AppState>,
    payload: web::Json<GenerateTextRequest>,
) -> impl Responder {
    println!("Generating text for prompt: {}", payload.prompt);
    let llama_model = &state.llama_model;

    // Generate text using LLaMA model
    match llama_model.generate_text(&payload.prompt, payload.max_length) {
        Ok(generated_text) => HttpResponse::Ok().json(GenerateTextResponse { generated_text }),
        Err(_) => HttpResponse::InternalServerError().body("Failed to generate text"),
    }
}

// #[post("/generate_text_stream")]
// pub async fn generate_text_stream(
//     state: web::Data<AppState>,
//     payload: web::Json<GenerateTextRequest>,
// ) -> impl Responder {
//     let llama_model = Arc::clone(&state.llama_model);
//     let prompt = payload.prompt.clone();
//     let max_length = payload.max_length;

//     type MsgResult = Result<sse::Event, Error>;
//     let (tx, rx) = mpsc::channel::<MsgResult>(100);

//     tokio::spawn(async move {
//         match llama_model.generate_text_stream(&prompt, max_length) {
//             Ok(stream) => {
//                 let mut pinned_stream = Box::pin(stream);

//                 while let Some(result) = pinned_stream.next().await {
//                     match result {
//                         Ok(word) => {
//                             if tx
//                                 .send(Ok(sse::Event::Data(sse::Data::new(word))))
//                                 .await
//                                 .is_err()
//                             {
//                                 break;
//                             }
//                         }
//                         Err(e) => {
//                             let _ = tx.send(Err(e.into())).await;
//                             break;
//                         }
//                     }
//                 }
//             }
//             Err(e) => {
//                 let _ = tx.send(Err(e.into())).await;
//             }
//         }
//     });

//     sse::Sse::from_receiver(rx).with_keep_alive(Duration::from_secs(5))
// }

// pub async fn generate_text_stream(
//     state: web::Data<AppState>,
//     payload: web::Json<GenerateTextRequest>,
// ) -> impl Responder {
//     println!("Streaming text for prompt: {}", payload.prompt);
//     let llama_model = &state.llama_model;

//     // Call the generate_text_stream function to get the stream
//     match llama_model.generate_text_stream(&payload.prompt, payload.max_length) {
//         Ok(token_stream) => {
//             let event_stream = token_stream.map(|result| match result {
//                 Ok(token) => {
//                     // Convert the string token into Bytes
//                     let message = format!("data: {}\n\n", token);
//                     Ok::<_, actix_web::Error>(Bytes::from(message))
//                 }
//                 Err(e) => {
//                     eprintln!("Error generating token: {:?}", e);
//                     Ok::<_, actix_web::Error>(Bytes::from("data: [Error]\n\n"))
//                 }
//             });

//             HttpResponse::Ok()
//                 .content_type("text/event-stream")
//                 .insert_header(("X-Accel-Buffering", HeaderValue::from_static("no")))
//                 .streaming(event_stream)
//         }
//         Err(e) => {
//             eprintln!("Failed to create stream: {:?}", e);
//             HttpResponse::InternalServerError().body("Failed to create text stream")
//         }
//     }
// }

pub async fn generate_text_stream(
    state: web::Data<AppState>,
    payload: web::Json<GenerateTextRequest>,
) -> impl Responder {
    println!("Streaming text for prompt: {}", payload.prompt);
    let llama_model = Arc::clone(&state.llama_model);

    // Encode the prompt and create the initial tokens
    let initial_tokens = match llama_model.encode_prompt(&payload.prompt) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Error encoding prompt: {:?}", e);
            return HttpResponse::InternalServerError().body("Failed to encode prompt");
        }
    };
    println!("Initial Tokens: {:?}", initial_tokens);

    let eos_token_id = llama_model.eos_token_id();
    println!("EOS Token ID: {:?}", eos_token_id);

    //TODO: remove this interval
    let mut interval = interval(Duration::from_secs(1));
    let stream = async_stream::stream! {
        let mut tokens = initial_tokens;
        let mut cache = match llama_model.create_cache() {
            Ok(cache) => cache,
            Err(e) => {
                yield Ok::<_, actix_web::Error>(Bytes::from("Error creating cache\n\n"));
                return;
            }
        };
        let mut logits_processor = llama_model.create_logits_processor(42);
        let mut index = 0;
        let mut index_pos = 0;

        loop {
            // Immediately yield a hardcoded message for debugging
            // interval.tick().await;
            // let data = format!("data: {}\n\n", "This is a hardcoded message");
            // println!("@@@Data: {}", data);
            // yield Ok::<_, actix_web::Error>(Bytes::from(data));

            if index >= payload.max_length {
                break;
            }

            let (context_size, context_index) = if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

            // Generate the next token
            let next_token = match llama_model.generate_next_token(ctxt, context_index, &mut cache, &mut logits_processor) {
                Ok(token) => token,
                Err(e) => {
                    eprintln!("Error generating next token: {:?}", e);
                    yield Ok::<_, actix_web::Error>(Bytes::from("Error generating next token\n\n"));
                    break;
                }
            };

            // Process the next token
            tokens.push(next_token);

            // Check for EOS token
            if Some(next_token) == eos_token_id {
                println!("EOS token found");
                yield Ok::<_, actix_web::Error>(Bytes::from("data: EOS\n\n"));
                break;
            }

            let formatted_text = llama_model.decode_token(next_token);
            println!("Generated token: {}", formatted_text);

            let message = format!("data: {}\n\n\n", formatted_text);
            println!("Message: {}", message);
            yield Ok::<_, actix_web::Error>(Bytes::from(message));
            // Explicitly flush the response
            yield Ok::<_, actix_web::Error>(Bytes::new());

            index += 1;
            index_pos += context_size;
        }

    };

    let boxed_stream: Pin<Box<dyn Stream<Item = Result<Bytes, actix_web::Error>>>> =
        Box::pin(stream);

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Content-Encoding", "identity"))
        .insert_header(("Cache-Control", "no-transform"))
        .insert_header(("X-Content-Type-Options", "nosniff"))
        .insert_header(("X-Accel-Buffering", HeaderValue::from_static("no")))
        .streaming(boxed_stream)
}
