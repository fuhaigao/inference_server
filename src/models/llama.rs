use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as model;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use model::{Config, Llama, LlamaConfig};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";

pub struct LlamaInferenceModel {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
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
        // using false to disable flash_attn optimization, which apple chip does not support
        let config = config.into_config(false);

        // Load weights
        let filenames = [
            // "model-00001-of-00002.safetensors",
            // "model-00002-of-00002.safetensors",
            "model.safetensors",
        ]
        .iter()
        .map(|&filename| repo.get(filename))
        .collect::<Result<Vec<_>, _>>()?;

        // Load the model
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
        let model = Llama::load(vb, &config)?;

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn generate_text(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        // Use cache to speed up the generation, first parameter 'true' means to use key-value cache
        let mut cache = model::Cache::new(true, DType::F32, &self.config, &self.device)?;
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
        // let temperature = Some(0.8);
        // let top_p = Some(0.7);
        // let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        // Use simple sampling
        let mut logits_processor = LogitsProcessor::new(seed, None, None);
        let start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut generated_text = String::new();

        for index in (0..max_length) {
            // Determine the context size and index for the current token
            let (context_size, context_index) = if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self
                .model
                .forward(&input, context_index, &mut cache)?
                .squeeze(0)?;

            // Update the index position
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let formatted_text = text.replace('▁', " ").replace("<0x0A>", "\n");
                generated_text.push_str(&formatted_text);
            }

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
