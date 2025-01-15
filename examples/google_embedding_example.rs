// Import required builder types from rllm
use rllm::builder::{LLMBackend, LLMBuilder};

/// Example demonstrating how to generate embeddings using Google's API
///
/// This example shows how to:
/// - Configure a Google LLM provider
/// - Generate embeddings for text input
/// - Access and display the resulting embedding vector
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the LLM builder with Google configuration
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google)
        // Get API key from environment variable or use test key
        .api_key(std::env::var("GOOGLE_API_KEY").unwrap_or("YOUR-TEST-KEY".to_string()))
        // Use Google's text embedding model
        .model("text-embedding-004")
        .build()?;

    // Generate embedding vector for sample text
    let vector = llm.embed(vec!["Hello world!".to_string()])?;

    // Print embedding statistics and data
    println!("Data: {:?}", &vector);

    Ok(())
}
