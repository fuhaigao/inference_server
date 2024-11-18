use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as model;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use model::{Config, Llama, LlamaConfig};
use std::pin::Pin;
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

    // pub fn generate_text_stream(
    //     &self,
    //     prompt: &str,
    //     max_length: usize,
    // ) -> anyhow::Result<BoxStream<'static, Result<String, anyhow::Error>>> {
    //     let cache = model::Cache::new(true, DType::F32, &self.config, &self.device)?;
    //     let initial_tokens = self
    //         .tokenizer
    //         .encode(prompt, true)
    //         .map_err(|e| anyhow::anyhow!(e))?
    //         .get_ids()
    //         .to_vec();
    //     println!("Initial Tokens: {:?}", initial_tokens);

    //     let eos_token_id = self.tokenizer.token_to_id("</s>");
    //     println!("EOS Token ID: {:?}", eos_token_id);

    //     let seed = 42;
    //     let logits_processor = LogitsProcessor::new(seed, None, None);
    //     let device = self.device.clone();
    //     let model = self.model.clone();
    //     let tokenizer = self.tokenizer.clone();

    //     // Create a stream using `stream::unfold`
    //     let stream = stream::unfold(
    //         (initial_tokens, cache, logits_processor, 0, 0), // tokens, cache, logits_processor, index, index_pos
    //         move |(mut tokens, mut cache, mut logits_processor, index, mut index_pos)| {
    //             let device = device.clone();
    //             let model = model.clone();
    //             let tokenizer = tokenizer.clone();

    //             async move {
    //                 if index >= max_length {
    //                     return None;
    //                 }

    //                 // Determine context size and context index
    //                 let (context_size, context_index) = if index > 0 {
    //                     (1, index_pos) // Single token for subsequent steps
    //                 } else {
    //                     (tokens.len(), 0) // Full context for the first step
    //                 };

    //                 let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
    //                 let input = Tensor::new(ctxt, &device).ok()?.unsqueeze(0).ok()?;

    //                 // Perform forward pass
    //                 let logits = model
    //                     .forward(&input, context_index, &mut cache)
    //                     .ok()?
    //                     .squeeze(0)
    //                     .ok()?;

    //                 // Update index_pos with the context size
    //                 index_pos += ctxt.len();

    //                 // Sample the next token
    //                 let next_token = logits_processor.sample(&logits).ok()?;
    //                 tokens.push(next_token);

    //                 // Check for EOS token
    //                 if Some(next_token) == eos_token_id {
    //                     println!("EOS token found");
    //                     return Some((
    //                         Ok("EOS".to_string()),
    //                         (tokens, cache, logits_processor, index + 1, index_pos),
    //                     ));
    //                 }

    //                 // Decode the next token into text
    //                 let formatted_text = tokenizer
    //                     .id_to_token(next_token)
    //                     .unwrap_or_default()
    //                     .replace('▁', " ")
    //                     .replace("<0x0A>", "\n");
    //                 println!("Generated token: {}", formatted_text);

    //                 Some((
    //                     Ok(formatted_text),
    //                     (tokens, cache, logits_processor, index + 1, index_pos),
    //                 ))
    //             }
    //         },
    //     );

    //     Ok(Box::pin(stream))
    // }

    pub fn generate_text_stream(
        &self,
        prompt: &str,
        max_length: usize,
    ) -> anyhow::Result<BoxStream<'static, Result<String, anyhow::Error>>> {
        let cache = model::Cache::new(true, DType::F32, &self.config, &self.device)?;
        let initial_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();
        println!("Initial Tokens: {:?}", initial_tokens);

        let eos_token_id = self.tokenizer.token_to_id("</s>");
        println!("EOS Token ID: {:?}", eos_token_id);

        let seed = 42;
        let logits_processor = LogitsProcessor::new(seed, None, None);
        let device = self.device.clone();
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        // Create a stream
        let stream = async_stream::stream! {
            let mut tokens = initial_tokens;
            let mut cache = cache;
            let mut logits_processor = logits_processor;
            let mut index = 0;
            let mut index_pos = 0;

            loop {
                if index >= max_length {
                    break;
                }

                // Determine context size and context index
                let (context_size, context_index) = if index > 0 {
                    (1, index_pos) // Single token for subsequent steps
                } else {
                    (tokens.len(), 0) // Full context for the first step
                };

                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = match Tensor::new(ctxt, &device)
                    .ok()
                    .and_then(|t| t.unsqueeze(0).ok())
                {
                    Some(input) => input,
                    None => {
                        yield Err(anyhow::anyhow!("Failed to create input tensor"));
                        break;
                    }
                };

                // Perform forward pass
                let logits = match model
                    .forward(&input, context_index, &mut cache)
                    .ok()
                    .and_then(|t| t.squeeze(0).ok())
                {
                    Some(logits) => logits,
                    None => {
                        yield Err(anyhow::anyhow!("Failed to perform forward pass"));
                        break;
                    }
                };

                // Update index_pos with the context size
                index_pos += ctxt.len();

                // Sample the next token
                let next_token = match logits_processor.sample(&logits).ok() {
                    Some(token) => token,
                    None => {
                        yield Err(anyhow::anyhow!("Failed to sample next token"));
                        break;
                    }
                };
                tokens.push(next_token);

                // Check for EOS token
                if Some(next_token) == eos_token_id {
                    println!("EOS token found");
                    yield Ok("EOS".to_string());
                    break;
                }

                // Decode the next token into text
                let formatted_text = tokenizer
                    .id_to_token(next_token)
                    .unwrap_or_default()
                    .replace('▁', " ")
                    .replace("<0x0A>", "\n");
                println!("Generated token: {}", formatted_text);

                yield Ok(formatted_text);

                index += 1;
            }
        };

        Ok(Box::pin(stream))
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

    // Method to perform a forward pass and return the next token
    pub fn generate_next_token(
        &self,
        ctxt: &[u32],
        context_index: usize,
        cache: &mut model::Cache,
        logits_processor: &mut LogitsProcessor,
    ) -> anyhow::Result<u32> {
        let input = Tensor::new(ctxt, &self.device)
            .ok()
            .and_then(|t| t.unsqueeze(0).ok())
            .ok_or_else(|| anyhow::anyhow!("Failed to create input tensor"))?;

        let logits = self
            .model
            .forward(&input, context_index, cache)
            .ok()
            .and_then(|t| t.squeeze(0).ok())
            .ok_or_else(|| anyhow::anyhow!("Failed to perform forward pass"))?;

        let next_token = logits_processor.sample(&logits)?;
        return Ok(next_token);
        // logits_processor
        //     .sample(&logits)
        //     .ok_or_else(|| anyhow::anyhow!("Failed to sample next token"))
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
