// Web server entry point
use crate::state::AppState;
use actix_web::{web, App, HttpServer};
use candle::Device;
use models::bert::BertInferenceModel;
use std::sync::Arc;

mod models;
mod state;

#[actix_web::main]
fn main() {
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "embeddings.bin",
        "my_embedding",
        Device::Cpu,
    )
    .expect("Failed to load BertInferenceModel");

    let shared_state = Arc::new(AppState {
        bert_model,
        // Add other models here as needed.
    });

    // HttpServer::new(move || {
    //     App::new()
    //         .app_data(web::Data::new(shared_state.clone()))
    //         .service(infer_sentence)
    //         .service(infer_embedding)
    // })
    // .bind(("0.0.0.0", 8080))?
    // .run()
    // .await
}
