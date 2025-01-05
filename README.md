# RLLM

**RLLM** is a **Rust** library that lets you use **multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/jmorganca/ollama), etc.  
With a **unified API** and **builder style** - similar to the Stripe experience - you can easily create **chat** or text **completion** requests without multiplying structures and crates.

## Key Features

- **Multi-backend**: Manage OpenAI, Anthropic, Ollama... through a single entry point.
- **Builder pattern**: Configure your LLM (model, temperature, max_tokens, timeouts...) with a few simple calls.
- **Chat & Completions**: Two unified traits (`ChatProvider` and `CompletionProvider`) to cover most use cases.
- **Extensible**: Easily add new backends.
- **Rust-friendly**: Designed with clear traits, unified error handling, and conditional compilation via *features*.

## Installation

Simply add **RLLM** to your `Cargo.toml`:

## Examples

| Name | Description |
|------|-------------|
| `anthropic_example` | Demonstrates integration with Anthropic's Claude model for chat completion |
| `chain_example` | Shows how to create multi-step prompt chains for exploring programming language features |
| `multi_backend_example` | Illustrates chaining multiple LLM backends (OpenAI, Anthropic) together in a single workflow |
| `ollama_example` | Example of using local LLMs through Ollama integration |
| `openai_example` | Basic OpenAI chat completion example with GPT models |

## Usage

```rust
use rllm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole},
};

fn main() {
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // or LLMBackend::Anthropic, LLMBackend::Ollama, etc.
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-4o")
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
