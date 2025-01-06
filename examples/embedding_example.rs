// Import required builder types from rllm
use rllm::builder::{LLMBackend, LLMBuilder};

/// Example demonstrating how to generate embeddings using OpenAI's API
///
/// This example shows how to:
/// - Configure an OpenAI LLM provider
/// - Generate embeddings for text input
/// - Access and display the resulting embedding vector
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the LLM builder with OpenAI configuration
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // .backend(LLMBackend::Ollama) or .backend(LLMBackend::XAI)
        // Get API key from environment variable or use test key
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".to_string()))
        // Use OpenAI's text embedding model
        .model("text-embedding-ada-002") // .model("v1") or .model("all-minilm")
        // Optional: Uncomment to customize embedding format and dimensions
        // .embedding_encoding_format("base64")
        // .embedding_dimensions(1536)
        .build()?;

    // Generate embedding vector for sample text
    let vector = llm.embed(vec!["Hello world!".to_string()])?;
    
    // Print embedding statistics and data
    println!("Data: {:?}", &vector);

    Ok(())
}
