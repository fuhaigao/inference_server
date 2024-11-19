use crate::state::AppState;
use actix_web::http::header::{HeaderValue, CACHE_CONTROL};
use actix_web::HttpRequest;
use actix_web::{web, HttpResponse, Responder};
use actix_web_lab::sse::{self, Data, Event};
use inference_server::models::llama::generate_next_token;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

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

pub async fn generate_text_stream(
    req: HttpRequest,
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

    // Create a channel for sending SSE events
    let (tx, rx) = tokio::sync::mpsc::channel(10);

    // Spawn a task to generate tokens and send them as SSE events
    actix_web::rt::spawn(async move {
        let mut tokens = initial_tokens;
        // Initialize cache
        let cache = match llama_model.create_cache() {
            Ok(c) => Arc::new(Mutex::new(c)),
            Err(e) => {
                eprintln!("Error creating cache: {:?}", e);
                // Handle the error appropriately, e.g., send an error message through the channel
                if let Err(send_err) = tx.send(Event::Comment("Error creating cache".into())).await
                {
                    eprintln!("Failed to send error message: {:?}", send_err);
                }
                return; // Exit the async block early
            }
        };

        // Initialize logits_processor
        let logits_processor = Arc::new(Mutex::new(llama_model.create_logits_processor(42)));

        let mut index = 0;
        let mut index_pos = 0;

        loop {
            if index >= payload.max_length {
                break;
            }

            let (context_size, context_index) = if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

            let next_token = match generate_next_token(
                Arc::clone(&llama_model),
                ctxt.to_vec(),
                context_index,
                Arc::clone(&cache),
                Arc::clone(&logits_processor),
            )
            .await
            {
                Ok(token) => token,
                Err(e) => {
                    eprintln!("Error generating next token: {:?}", e);
                    let _ = tx
                        .send(Event::Comment("Error generating next token".into()))
                        .await;
                    break;
                }
            };

            // Process the next token
            tokens.push(next_token);

            // Check for EOS token
            if Some(next_token) == eos_token_id {
                println!("EOS token found");
                let _ = tx.send(Event::Data(Data::new("EOS"))).await;
                break;
            }

            let formatted_text = llama_model.decode_token(next_token);
            println!("Generated token: {:?}", formatted_text);

            let message = Event::Data(Data::new(formatted_text));
            if tx.send(message).await.is_err() {
                // Client disconnected
                break;
            }

            index += 1;
            index_pos += context_size;
        }
    });

    // Return the SSE response
    sse::Sse::from_infallible_receiver(rx)
        .with_keep_alive(Duration::from_secs(10)) // Optional: send keep-alive comments every 10 seconds
        .customize()
        .insert_header((CACHE_CONTROL, HeaderValue::from_static("no-transform")))
        .insert_header(("X-Accel-Buffering", "no"))
        .respond_to(&req)
        .map_into_boxed_body()
}
