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
        embeddings_file: &str,
        embeddings_key: &str,
        device: Device,
    ) -> anyhow::Result<Self> {
        let embedding_tensor = match embeddings_file.is_empty() {
            true => {
                println!("no file name provided, embaddings return an empty tensor");
                Tensor::new(&[0.0], &device)?
            }
            false => {
                let tensor_file = safetensors::load(embeddings_file, &device)?;
                tensor_file
                    .get(embeddings_key)
                    .expect("Failed to retrieve embedding key")
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
        let api = Api::new()?.repo(repo);
        let config_path = api.get("config.json")?;
        let tokenizer_path = api.get("tokenizer.json")?;
        let weights_path = api.get("model.safetensors")?;

        // load the model config
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        //load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

        // load the model
        let variable_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
        let model = BertModel::load(variable_builder, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_tensor,
        })
    }

    pub fn infer_sentence_embedding(&self, sentence: &str) -> anyhow::Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode(sentence, true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        let pooled_embedding = Self::apply_max_pooling(&embeddings)?;
        Ok(Self::l2_normalize(&pooled_embedding)?)
    }

    pub fn create_embeddings(&self, sentences: Vec<String>) -> anyhow::Result<Tensor> {
        println!("Generating embeddings for {} sentences", sentences.len());
        let tokens = self
            .tokenizer
            .encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let token_vec = tokens.get_ids().to_vec();
                Ok(Tensor::new(token_vec.as_slice(), &self.device)?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        println!("Stacking tokens completed");

        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        let pooled_embeddings = Self::apply_max_pooling(&embeddings)?; // apply pooling (avg or max
        Ok(Self::l2_normalize(&pooled_embeddings)?)
    }

    pub fn score_vector_similarity(
        &self,
        query_vector: Tensor,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let vec_len = self.embedding_tensor.dim(0)?;
        let mut scores = vec![(0, 0.0); vec_len];
        for (embedding_index, score_tuple) in scores.iter_mut().enumerate() {
            let current_embedding = self.embedding_tensor.get(embedding_index)?.unsqueeze(0)?;
            // because its normalized we can use cosine similarity
            let cosine_similarity = (&current_embedding * &query_vector)?
                .sum_all()?
                .to_scalar::<f32>()?;
            *score_tuple = (embedding_index, cosine_similarity);
        }
        // sort scores by cosine_similarity
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        Ok(scores)
    }

    pub fn apply_max_pooling(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        Ok(embeddings.max(1)?)
    }

    pub fn l2_normalize(embeddings: &Tensor) -> anyhow::Result<Tensor> {
        Ok(embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
