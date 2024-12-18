// Web server entry point
use crate::api::{find_similar, generate_text, generate_text_stream};
use crate::state::AppState;
use actix_files as fs;
use actix_web::{web, App, HttpServer};
use candle::Device;
use env_logger;
use inference_server::models::bert::BertInferenceModel;
use inference_server::models::llama::LlamaInferenceModel;
use std::fs::File;
use std::sync::Arc;

mod api;
mod models;
mod state;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    // Load the BERT model
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        // "main",
        "refs/pr/21",
        "embeddings.bin",
        "my_embedding",
        Device::Cpu,
    )
    .expect("Failed to load BertInferenceModel");
    println!("Loaded BERT model");

    // Load text map from binary file
    let mut text_map_file = File::open("text_map.bin").expect("Failed to open text_map.bin");
    let text_map: Vec<String> =
        bincode::decode_from_std_read(&mut text_map_file, bincode::config::standard())
            .expect("Failed to decode text_map.bin");

    let llama_model = LlamaInferenceModel::load_from_hub(
        // "meta-llama/Llama-2-7b-chat-hf",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        Device::Cpu,
        Some("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".to_string()),
        None,
    )
    .expect("Failed to load LLAMA model");
    println!("Loaded LLAMA model");

    // Set up shared application state
    let shared_state = AppState {
        bert_model: Arc::new(bert_model),
        text_map,
        llama_model: Arc::new(llama_model),
    };

    // Start the HTTP server
    println!("Starting HTTP server on 0.0.0.0:8080...");
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(shared_state.clone()))
            .route("/find_similar", web::post().to(find_similar)) // API endpoint for finding similar topics
            .route("/generate_text", web::post().to(generate_text)) // API endpoint for generating text (non-streaming)
            .route(
                "/generate_text_stream",
                web::post().to(generate_text_stream),
            )
            .service(fs::Files::new("/", "./frontend/build").index_file("index.html"))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
