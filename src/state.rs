// Shared state management for models
use crate::models::bert::BertInferenceModel;
use std::sync::Arc;

pub struct AppState {
    pub bert_model: BertInferenceModel,
}
