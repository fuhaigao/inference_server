use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as model;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use model::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";

pub struct LlamaInferenceModel {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
}

impl LlamaInferenceModel {
    pub fn load_from_hub(
        model_id: &str,
        device: Device,
        huggingface_token: Option<String>,
        revision: Option<&str>,
    ) -> anyhow::Result<Self> {
        // Configure the API client
        let api = ApiBuilder::new().with_token(huggingface_token).build()?;
        let revision = revision.unwrap_or("main");

        // Set up the repo and retrieve config and tokenizer files
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;

        // Parse LLaMA configuration
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        // false to disable flash_attn optimization, which apple chip does not support
        let config = config.into_config(false);

        // Load weights
        let filenames = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            // "model.safetensors",
        ]
        .iter()
        .map(|&filename| repo.get(filename))
        .collect::<Result<Vec<_>, _>>()?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
        let cache = model::Cache::new(true, DType::F32, &config, &device)?;
        let model = Llama::load(vb, &cache, &config)?;

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn generate_text(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();
        println!("Tokens: {:?}", tokens);

        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);
        println!("EOS token ID: {:?}", eos_token_id);

        let seed = 42;
        let mut logits_processor = LogitsProcessor::new(seed, Some(1.0), Some(0.9)); // Wrapping values in Some()
        let start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut generated_text = String::new();

        for index in 0..max_length {
            // Limit the context size for each forward pass
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Forward pass through the model
            let logits = self.model.forward(&input, index_pos)?.squeeze(0)?;
            index_pos += ctxt.len();

            // Sample the next token
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            // Convert token to text and append it to the generated text
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let formatted_text = text.replace('▁', " ").replace("<0x0A>", "\n");
                generated_text.push_str(&formatted_text);
            }

            // Check for end-of-sequence token
            if Some(next_token) == eos_token_id {
                println!("EOS token found");
                break;
            }
        }

        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            tokens.len(),
            tokens.len() as f64 / dt.as_secs_f64(),
        );
        println!("Generated text: {:?}", generated_text);

        Ok(generated_text)
    }
}
