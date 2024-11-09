// Shared state management for models
use inference_server::models::bert::BertInferenceModel;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub bert_model: Arc<BertInferenceModel>,
    pub text_map: Vec<String>,
}
