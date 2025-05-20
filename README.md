# RLLM

> Note: Starting with version 1.x, RLLM has become a simple wrapper around [llm](https://github.com/graniet/llm).
> Both crates will be actively maintained and kept in sync.
> If you are new to this ecosystem, you can use either [llm](https://github.com/graniet/llm) directly or rllm - they provide the same features.

**RLLM** is a **Rust** library that lets you use **multiple LLM backends** in a single project:  [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com), [Groq](https://www.groq.com) and [Google](https://cloud.google.com/gemini).
With a **unified API** and **builder style** - similar to the Stripe experience - you can easily create **chat** or text **completion** requests without multiplying structures and crates.

### Base crate is :  [LLM](https://github.com/graniet/llm) 

## Key Features

- **Multi-backend**: Manage OpenAI, Anthropic, Ollama, DeepSeek, xAI, Phind, Groq and Google through a single entry point.
- **Multi-step chains**: Create multi-step chains with different backends at each step.
- **Templates**: Use templates to create complex prompts with variables.
- **Builder pattern**: Configure your LLM (model, temperature, max_tokens, timeouts...) with a few simple calls.
- **Chat & Completions**: Two unified traits (`ChatProvider` and `CompletionProvider`) to cover most use cases.
- **Extensible**: Easily add new backends.
- **Rust-friendly**: Designed with clear traits, unified error handling, and conditional compilation via *features*.
- **Validation**: Add validation to your requests to ensure the output is what you expect.
- **Evaluation**: Add evaluation to your requests to score the output of LLMs.
- **Parallel Evaluation**: Evaluate multiple LLM providers in parallel and select the best response based on scoring functions.
- **Function calling**: Add function calling to your requests to use tools in your LLMs.
- **REST API**: Serve any LLM backend as a REST API with openai standard format.
- **Vision**: Add vision to your requests to use images in your LLMs.
- **Reasoning**: Add reasoning to your requests to use reasoning in your LLMs.
- **Structured Output**: Request structured output from certain LLM providers based on a provided JSON schema.
- **Speech to text**: Transcribe audio to text

## Examples

Go to [LLM Examples](https://github.com/graniet/llm/tree/main/examples)
