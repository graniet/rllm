// Import required modules from the RLLM library for Phind integration
use rllm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::{ChatMessage, ChatRole},     // Chat-related structures
};

fn main() {
    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Phind) // Use Phind as the LLM provider
        .model("Phind-70B")        // Use Phind-70B model
        .max_tokens(512)           // Limit response length
        .temperature(0.7)          // Control response randomness (0.0-1.0)
        .stream(false)             // Disable streaming responses
        .build()
        .expect("Failed to build LLM (Phind)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love cats".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content: "I am an assistant, I cannot love cats but I can love dogs".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love dogs in 2000 chars".into(),
        },
    ];

    // Send chat request and handle the response
    match llm.chat(&messages) {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }
}
