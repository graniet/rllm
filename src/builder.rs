/// Module for building and configuring LLM providers.
use crate::{error::RllmError, LLMProvider};

/// Supported LLM backend providers.
#[derive(Debug, Clone)]
pub enum LLMBackend {
    /// OpenAI API provider (GPT models)
    OpenAI,
    /// Anthropic API provider (Claude models)
    Anthropic,
    /// Ollama local LLM provider
    Ollama,
    /// DeepSeek API provider (LLM models)
    DeepSeek,
}

/// Builder for configuring and instantiating LLM providers.
#[derive(Debug, Default)]
pub struct LLMBuilder {
    /// Selected backend provider
    backend: Option<LLMBackend>,
    /// API key for authentication
    api_key: Option<String>,
    /// Base URL for API requests
    base_url: Option<String>,
    /// Model identifier to use
    model: Option<String>,
    /// Maximum tokens to generate
    max_tokens: Option<u32>,
    /// Temperature for controlling randomness
    temperature: Option<f32>,
    /// System prompt/context
    system: Option<String>,
    /// Request timeout in seconds
    timeout_seconds: Option<u64>,
    /// Whether to enable streaming responses
    stream: Option<bool>,
    /// Top p for controlling randomness
    top_p: Option<f32>,
    /// Top k for controlling randomness
    top_k: Option<u32>,
}

impl LLMBuilder {
    /// Creates a new empty builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the backend provider to use.
    pub fn backend(mut self, backend: LLMBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Sets the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL for API requests.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the model identifier to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the temperature for controlling response randomness.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the system prompt/context.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the request timeout in seconds.
    pub fn timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Enables or disables streaming responses.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Sets the top p for controlling randomness.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the top k for controlling randomness.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Builds and returns a configured LLM provider instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No backend is specified
    /// - Required backend feature is not enabled
    /// - Required configuration like API keys are missing
    pub fn build(self) -> Result<Box<dyn LLMProvider>, RllmError> {
        let backend = self
            .backend
            .ok_or_else(|| RllmError::InvalidRequest("No backend specified".to_string()))?;

        match backend {
            LLMBackend::OpenAI => {
                #[cfg(not(feature = "openai"))]
                {
                    Err(RllmError::InvalidRequest(
                        "OpenAI feature not enabled".to_string(),
                    ))
                }
                #[cfg(feature = "openai")]
                {
                    let key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for OpenAI".to_string())
                    })?;
                    let openai = crate::backends::openai::OpenAI::new(
                        key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                    );
                    Ok(Box::new(openai) as Box<dyn LLMProvider>)
                }
            }
            LLMBackend::Anthropic => {
                #[cfg(not(feature = "anthropic"))]
                {
                    Err(RllmError::InvalidRequest(
                        "Anthropic feature not enabled".to_string(),
                    ))
                }
                #[cfg(feature = "anthropic")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for Anthropic".to_string())
                    })?;

                    let anthro = crate::backends::anthropic::Anthropic::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                    );
                    impl crate::LLMProvider for crate::backends::anthropic::Anthropic {}
                    Ok(Box::new(anthro) as Box<dyn LLMProvider>)
                }
            }
            LLMBackend::Ollama => {
                #[cfg(not(feature = "ollama"))]
                {
                    Err(RllmError::InvalidRequest(
                        "Ollama feature not enabled".to_string(),
                    ))
                }
                #[cfg(feature = "ollama")]
                {
                    let url = self
                        .base_url
                        .unwrap_or("http://localhost:11434".to_string());
                    let ollama = crate::backends::ollama::Ollama::new(
                        url,
                        self.api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                    );
                    impl crate::LLMProvider for crate::backends::ollama::Ollama {}
                    Ok(Box::new(ollama) as Box<dyn LLMProvider>)
                }
            }
            LLMBackend::DeepSeek => {
                #[cfg(not(feature = "deepseek"))]
                {
                    Err(RllmError::InvalidRequest("DeepSeek feature not enabled".to_string()))
                }

                #[cfg(feature = "deepseek")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for DeepSeek".to_string())
                    })?;

                    let deepseek = crate::backends::deepseek::DeepSeek::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                    );

                    Ok(Box::new(deepseek) as Box<dyn LLMProvider>)
                }
            }
        }
    }
}
