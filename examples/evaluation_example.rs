// Import required modules from RLLM and serde_json
use rllm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole}, 
    evaluator::{EvalResult, LLMEvaluator},
};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenAI LLM with API key from environment or fallback
    let openai = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .model("gpt-3.5-turbo")
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-OPENAI".into()))
        .build()?;

    // Initialize Anthropic LLM with API key from environment or fallback
    let anthropic = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .model("claude-3-5-sonnet-20240620")
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthropic-key".into()))
        .build()?;

    // Initialize Phind LLM (no API key required)
    let phind = LLMBuilder::new()
        .backend(LLMBackend::Phind)
        .model("Phind-70B")
        .build()?;

    // Initialize DeepSeek LLM with API key from environment or fallback
    let deepseek = LLMBuilder::new()
        .backend(LLMBackend::DeepSeek)
        .model("deepseek-chat")
        .api_key(std::env::var("DEEPSEEK_API_KEY").unwrap_or("deepseek-key".into()))
        .build()?;

    // Create evaluator with all LLM providers and custom scoring function
    let evaluator = LLMEvaluator::new(vec![openai, anthropic, phind, deepseek])
        .scoring(|resp_text| {
            // Try to parse response as JSON
            let json_result = serde_json::from_str::<Value>(resp_text);
            
            match json_result {
                Ok(json) => {
                    let mut total_score = 0.0;
                    
                    // Score based on greeting field content
                    if let Some(greeting) = json.get("greeting") {
                        if let Some(text) = greeting.as_str() {
                            let text_lower = text.to_lowercase();
                            // Base point for having greeting field
                            total_score += 1.0;
                            // Points for each "yes" occurrence
                            total_score += text_lower.matches("yes").count() as f32;
                            // Bonus point for common greeting words
                            if text_lower.contains("hello") || text_lower.contains("hi") || text_lower.contains("hey") {
                                total_score += 1.0;
                            }
                        }
                    }
                    
                    total_score
                },
                Err(_) => 0.0 // Zero score for invalid JSON
            }
        });

    // Prepare chat message requesting JSON greeting
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "Give me a short greeting with only the word 'yes' in it! in json format with this format : {greeting: 'yes'}. only JSON.".into(),
    }];

    // Evaluate responses from all LLMs
    let results: Vec<EvalResult> = evaluator.evaluate_chat(&messages)?;
    // Print results with scores
    for (i, item) in results.iter().enumerate() {
        println!(
            "LLM #{} => Score={:.2}, text:\n{}",
            i, item.score, item.text
        );
    }

    Ok(())
}
