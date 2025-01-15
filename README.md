# RLLM

**RLLM** is a **Rust** library that lets you use **multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com) and [Google](https://cloud.google.com/gemini).
With a **unified API** and **builder style** - similar to the Stripe experience - you can easily create **chat** or text **completion** requests without multiplying structures and crates.

## Key Features

- **Multi-backend**: Manage OpenAI, Anthropic, Ollama, DeepSeek, xAI, Phind and Google through a single entry point.
- **Multi-step chains**: Create multi-step chains with different backends at each step.
- **Templates**: Use templates to create complex prompts with variables.
- **Builder pattern**: Configure your LLM (model, temperature, max_tokens, timeouts...) with a few simple calls.
- **Chat & Completions**: Two unified traits (`ChatProvider` and `CompletionProvider`) to cover most use cases.
- **Extensible**: Easily add new backends.
- **Rust-friendly**: Designed with clear traits, unified error handling, and conditional compilation via *features*.
- **Validation**: Add validation to your requests to ensure the output is what you expect.
- **Evaluation**: Add evaluation to your requests to score the output of LLMs.

## Installation

Simply add **RLLM** to your `Cargo.toml`:

```toml
[dependencies]
rllm = { version = "1.1.0", features = ["openai", "anthropic", "ollama"] }
```

## Examples

| Name | Description |
|------|-------------|
| [`anthropic_example`](examples/anthropic_example.rs) | Demonstrates integration with Anthropic's Claude model for chat completion |
| [`chain_example`](examples/chain_example.rs) | Shows how to create multi-step prompt chains for exploring programming language features |
| [`deepseek_example`](examples/deepseek_example.rs) | Basic DeepSeek chat completion example with deepseek-chat models |
| [`embedding_example`](examples/embedding_example.rs) | Basic embedding example with OpenAI's API |
| [`multi_backend_example`](examples/multi_backend_example.rs) | Illustrates chaining multiple LLM backends (OpenAI, Anthropic, DeepSeek) together in a single workflow |
| [`ollama_example`](examples/ollama_example.rs) | Example of using local LLMs through Ollama integration |
| [`openai_example`](examples/openai_example.rs) | Basic OpenAI chat completion example with GPT models |
| [`phind_example`](examples/phind_example.rs) | Basic Phind chat completion example with Phind-70B model |
| [`validator_example`](examples/validator_example.rs) | Basic validator example with Anthropic's Claude model |
| [`xai_example`](examples/xai_example.rs) | Basic xAI chat completion example with Grok models |
| [`evaluation_example`](examples/evaluation_example.rs) | Basic evaluation example with Anthropic, Phind and DeepSeek |
| [`google_example`](examples/google_example.rs) | Basic Google Gemini chat completion example with Gemini models |
| [`google_embedding_example`](examples/google_embedding_example.rs) | Basic Google Gemini embedding example with Gemini models |

## Usage
Here's a basic example using OpenAI for chat completion. See the examples directory for other backends (Anthropic, Ollama, DeepSeek, xAI, Google, Phind), embedding capabilities, and more advanced use cases.

```rust
use rllm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole},
};

fn main() {
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // or LLMBackend::Anthropic, LLMBackend::Ollama, LLMBackend::DeepSeek, LLMBackend::XAI, LLMBackend::Phind ...
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-4o") // or model("claude-3-5-sonnet-20240620") or model("grok-2-latest") or model("deepseek-chat") or model("llama3.1") or model("Phind-70B") ...
        .max_tokens(1000)
        .temperature(0.7)
        .system("You are a helpful assistant.")
        .stream(false)
        .build()
        .expect("Failed to build LLM");
}

    let messages = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love cats".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content:
                "I am an assistant, I cannot love cats but I can love dogs"
                    .into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love dogs in 2000 chars".into(),
        },
    ];

    let chat_resp = llm.chat(&messages);
    match chat_resp {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }
```
