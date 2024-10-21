// BERT model
use candle::{safetensors, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub struct BertInferenceModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    embedding_tensor: Tensor,
}

impl BertInferenceModel {
    pub fn load(
        model_name: &str,
        model_revision: &str,
        embeddings_filename: &str,
        embeddings_key: &str,
        device: Device,
    ) -> anyhow::Result<Self> {
        let embedding_tensor = match embeddings_filename.is_empty() {
            true => {
                // TODO: use log
                println!("no file name provided, embaddings return an empty tensor");
                Tensor::new(&[0.0], &device)?
            }
            false => {
                let tensor_file = safetensors::load(embeddings_filename, &device)?;
                tensor_file
                    .get(embeddings_key)
                    .expect("error getting key:embedding")
                    .clone()
            }
        };
        println!("loaded embedding shape: {:?}", embedding_tensor.shape());

        // start loading the model from the repo
        let repo = Repo::with_revision(
            model_name.parse()?,
            RepoType::Model,
            model_revision.parse()?,
        );
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model.safetensors")?;

        // load the model config
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        //load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        // load the model
        let variable_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(variable_builder, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_tensor,
        })
    }
}
