use actix_web::web;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as model;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use model::{Config, Llama, LlamaConfig};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";

pub struct LlamaInferenceModel {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: Config,
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
        let mut cache = self.create_cache()?;
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
        // let mut logits_processor = LogitsProcessor::new(seed, None, None);
        let mut logits_processor = self.create_logits_processor(seed);
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
            println!("next_token: {:?}", next_token);
            tokens.push(next_token);

            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let formatted_text = text.replace('▁', " ").replace("<0x0A>", "\n");
                println!("formatted_text: {:?}", formatted_text);
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

    // Method to encode the prompt into initial tokens
    pub fn encode_prompt(&self, prompt: &str) -> anyhow::Result<Vec<u32>> {
        self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))
            .map(|encoded| encoded.get_ids().to_vec())
    }

    // Method to create a cache
    pub fn create_cache(&self) -> anyhow::Result<model::Cache> {
        model::Cache::new(true, DType::F32, &self.config, &self.device)
            .map_err(|e| anyhow::anyhow!(e))
    }

    // Method to create a logits processor
    pub fn create_logits_processor(&self, seed: u64) -> LogitsProcessor {
        LogitsProcessor::new(seed, None, None)
    }

    // Method to decode a token ID into text
    pub fn decode_token(&self, token_id: u32) -> String {
        self.tokenizer
            .id_to_token(token_id)
            .unwrap_or_default()
            .replace('▁', " ")
            .replace("<0x0A>", "\n")
    }

    // Method to get the EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("</s>")
    }
}
/*
Asynchronously generates the next token using the provided model and context.

 # Arguments
 - `model`: An `Arc` to the `LlamaInferenceModel`. Ownership is taken to allow the model
   to be safely shared across threads during the blocking operation.
 - `ctxt`: A `Vec<u32>` representing the current context. Ownership is taken to transfer
   the data into the separate thread without requiring synchronization mechanisms.
 - `context_index`: The index within the context to process.
 - `cache`: An `Arc<Mutex<model::Cache>>` for caching model computations. Ownership is
   taken to ensure safe, concurrent access within the separate thread.
 - `logits_processor`: An `Arc<Mutex<LogitsProcessor>>` for processing logits. Ownership
   is taken to allow safe, concurrent processing within the separate thread.

 # Returns
 - `Ok(u32)`: The next token generated.
 - `Err(anyhow::Error)`: An error occurred during token generation.

 This function offloads the blocking operation to a separate thread using `web::block`.
 By taking ownership of the arguments, it ensures that all necessary data is moved into
 the new thread, preventing potential data races and ensuring thread safety.
*/
pub async fn generate_next_token(
    model: Arc<LlamaInferenceModel>,
    ctxt: Vec<u32>,
    context_index: usize,
    cache: Arc<Mutex<model::Cache>>,
    logits_processor: Arc<Mutex<LogitsProcessor>>,
) -> anyhow::Result<u32> {
    let device = model.device.clone();

    // Offload the blocking operation to a separate thread
    let next_token = web::block(move || {
        let input = Tensor::new(ctxt.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {:?}", e))?;

        let mut cache = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock cache: {:?}", e))?;

        let logits = model
            .model
            .forward(&input, context_index, &mut cache)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| anyhow::anyhow!("Failed to perform forward pass: {:?}", e))?;

        let mut logits_processor = logits_processor
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock logits processor: {:?}", e))?;

        logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("Failed to sample next token: {:?}", e))
    })
    .await??;

    Ok(next_token)
}
