use anyhow::Result;
use bincode;
use candle::Tensor;
use csv;
use inference_server::models::bert::BertInferenceModel;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;

fn main() -> Result<()> {
    // Get the file name from command-line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: embedding_generator <file_name>");
        std::process::exit(1);
    }
    let csv_file_path = &args[1];
    println!("Starting to generate embeddings from {}", csv_file_path);

    // Load sentences from the CSV file
    let text_map: HashMap<String, String> =
        load_sentences_from_csv(csv_file_path, 0, 1).expect("Failed to load sentences from CSV");
    println!("Loaded sentences - total count: {}", text_map.len());

    // Convert sentences to a Vec<String> for embedding generation
    let sentences: Vec<String> = text_map.values().map(|s| s.to_string()).collect();

    // Serialize the map to a binary file
    let mut sentence_file = File::create("text_map.bin")
        .expect("Failed to create text_map.bin for serialized sentences");
    bincode::encode_into_std_write(&sentences, &mut sentence_file, bincode::config::standard())
        .expect("Failed to encode sentences");
    println!("Serialized text_map to text_map.bin");

    // Load the BERT model
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        // "main",
        "refs/pr/21",
        "",
        "",
        candle::Device::Cpu,
    )
    .expect("Failed to load BERT model");
    println!("Loaded BERT model");

    // Generate embeddings in parallel
    let embedding_results: Vec<Result<Tensor, _>> = sentences
        .par_chunks(200)
        .map(|chunk| bert_model.create_embeddings(chunk.to_vec()))
        .collect();
    println!("Embeddings generated");

    // Concatenate the embeddings and save as a binary file
    let embeddings = Tensor::cat(
        &embedding_results
            .iter()
            .map(|res| res.as_ref().unwrap())
            .collect::<Vec<_>>(),
        0,
    )
    .expect("Failed to concatenate embeddings");
    embeddings
        .save_safetensors("my_embedding", "embeddings.bin")
        .expect("Failed to save embeddings.bin");
    println!("Saved embeddings.bin");

    Ok(())
}

// Helper function to load CSV data into a HashMap
fn load_sentences_from_csv(
    csv_file_path: &str,
    name_col_index: usize,
    text_col_index: usize,
) -> Result<HashMap<String, String>> {
    let mut sentence_map = HashMap::new();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_file_path)?;
    for result in reader.records() {
        let record = result?;
        let name = record.get(name_col_index).unwrap().to_string();
        let text = record.get(text_col_index).unwrap().to_string();
        sentence_map.insert(name, text);
    }
    Ok(sentence_map)
}
