// API routes and handlers
use crate::state::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

// #[derive(Deserialize)]
// pub struct InferenceRequest {
//     text: String,
//     top_k: usize,
// }

// #[derive(Serialize)]
// pub struct InferenceResponse {
//     results: Vec<String>,
// }

// #[post("/infer")]
// pub async fn infer_sentence(
//     state: web::Data<AppState>,
//     payload: web::Json<InferenceRequest>,
// ) -> impl Responder {
//     let bert_model = &state.bert_model;

//     match bert_model.infer_sentence_embedding(&payload.text) {
//         Ok(embedding) => {
//             let results = bert_model
//                 .score_vector_similarity(embedding, payload.top_k)
//                 .unwrap_or_default();

//             let response = InferenceResponse {
//                 results: results
//                     .into_iter()
//                     .map(|(index, score)| format!("Index: {}, Score: {}", index, score))
//                     .collect(),
//             };

//             HttpResponse::Ok().json(response)
//         },
//         Err(_) => HttpResponse::InternalServerError().body("Inference failed"),
//     }
// }

// #[post("/embedding")]
// pub async fn infer_embedding(
//     state: web::Data<AppState>,
//     payload: web::Json<InferenceRequest>,
// ) -> impl Responder {
//     let bert_model = &state.bert_model;

//     match bert_model.create_embeddings(vec![payload.text.clone()]) {
//         Ok(embedding) => {
//             HttpResponse::Ok().json(InferenceResponse {
//                 results: vec![format!("Embedding Shape: {:?}", embedding.shape())],
//             })
//         },
//         Err(_) => HttpResponse::InternalServerError().body("Embedding creation failed"),
//     }
// }
