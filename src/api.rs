// API routes and handlers
use crate::state::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

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
struct SimilarityRequest {
    text: String,
    num_results: usize,
}

#[derive(Deserialize)]
struct GenerateTextRequest {
    prompt: String,
    max_length: usize,
}

#[derive(Serialize)]
struct GenerateTextResponse {
    generated_text: String,
}

#[post("/find_similar")]
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

    let results: Vec<(usize, f32)> = bert_model
        .score_vector_similarity(query_embedding, payload.num_results as usize)
        .unwrap();

    let top_results: Vec<TopResult> = results
        .into_iter()
        .map(|record| {
            let top_item_text = text_map.get(record.0).unwrap();
            TopResult {
                item: top_item_text.to_string(),
                score: record.1,
            }
        })
        .collect();

    HttpResponse::Ok().json(SimilarityResponse { top_results })
}

#[post("/generate_text")]
pub async fn generate_text(
    state: web::Data<AppState>,
    payload: web::Json<GenerateTextRequest>,
) -> impl Responder {
    println!("Generating text for prompt: {}", payload.prompt);
    let llama_model = &state.llama_model;

    // Generate text using LLaMA model
    match llama_model.generate_text(&payload.prompt, payload.max_length) {
        Ok(generated_text) => HttpResponse::Ok().json(GenerateTextResponse { generated_text }),
        Err(e) => {
            println!("Error: {:?}", e);
            HttpResponse::InternalServerError().body("Failed to generate text")
        }
    }
}
