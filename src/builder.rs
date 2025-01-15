//! Builder module for configuring and instantiating LLM providers.
//!
//! This module provides a flexible builder pattern for creating and configuring
//! LLM (Large Language Model) provider instances with various settings and options.

use crate::{error::RllmError, LLMProvider};

/// A function type for validating LLM provider outputs.
/// Takes a response string and returns Ok(()) if valid, or Err with an error message if invalid.
pub type ValidatorFn = dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static;

/// Supported LLM backend providers.
#[derive(Debug, Clone)]
pub enum LLMBackend {
    /// OpenAI API provider (GPT-3, GPT-4, etc.)
    OpenAI,
    /// Anthropic API provider (Claude models)
    Anthropic,
    /// Ollama local LLM provider for self-hosted models
    Ollama,
    /// DeepSeek API provider for their LLM models
    DeepSeek,
    /// X.AI (formerly Twitter) API provider
    XAI,
    /// Phind API provider for code-specialized models
    Phind,
    /// Google Gemini API provider
    Google,
}

/// Implements string parsing for LLMBackend enum.
///
/// Converts a string representation of a backend provider name into the corresponding
/// LLMBackend variant. The parsing is case-insensitive.
///
/// # Arguments
///
/// * `s` - The string to parse
///
/// # Returns
///
/// * `Ok(LLMBackend)` - The corresponding backend variant if valid
/// * `Err(RllmError)` - An error if the string doesn't match any known backend
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
/// use rllm::LLMBackend;
///
/// let backend = LLMBackend::from_str("openai").unwrap();
/// assert!(matches!(backend, LLMBackend::OpenAI));
///
/// let err = LLMBackend::from_str("invalid").unwrap_err();
/// assert!(err.to_string().contains("Unknown LLM backend"));
/// ```
impl std::str::FromStr for LLMBackend {
    type Err = RllmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(LLMBackend::OpenAI),
            "anthropic" => Ok(LLMBackend::Anthropic), 
            "ollama" => Ok(LLMBackend::Ollama),
            "deepseek" => Ok(LLMBackend::DeepSeek),
            "xai" => Ok(LLMBackend::XAI),
            "phind" => Ok(LLMBackend::Phind),
            _ => Err(RllmError::InvalidRequest(format!(
                "Unknown LLM backend: {}",
                s
            ))),
        }
    }
}

/// Builder for configuring and instantiating LLM providers.
///
/// Provides a fluent interface for setting various configuration options
/// like model selection, API keys, generation parameters, etc.
#[derive(Default)]
pub struct LLMBuilder {
    /// Selected backend provider
    backend: Option<LLMBackend>,
    /// API key for authentication with the provider
    api_key: Option<String>,
    /// Base URL for API requests (primarily for self-hosted instances)
    base_url: Option<String>,
    /// Model identifier/name to use
    model: Option<String>,
    /// Maximum tokens to generate in responses
    max_tokens: Option<u32>,
    /// Temperature parameter for controlling response randomness (0.0-1.0)
    temperature: Option<f32>,
    /// System prompt/context to guide model behavior
    system: Option<String>,
    /// Request timeout duration in seconds
    timeout_seconds: Option<u64>,
    /// Whether to enable streaming responses
    stream: Option<bool>,
    /// Top-p (nucleus) sampling parameter
    top_p: Option<f32>,
    /// Top-k sampling parameter
    top_k: Option<u32>,
    /// Format specification for embedding outputs
    embedding_encoding_format: Option<String>,
    /// Vector dimensions for embedding outputs
    embedding_dimensions: Option<u32>,
    /// Optional validation function for response content
    validator: Option<Box<ValidatorFn>>,
    /// Number of retry attempts when validation fails
    validator_attempts: usize,
}

impl LLMBuilder {
    /// Creates a new empty builder instance with default values.
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

    /// Sets the temperature for controlling response randomness (0.0-1.0).
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

    /// Sets the top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the top-k sampling parameter.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets the encoding format for embeddings.
    pub fn embedding_encoding_format(
        mut self,
        embedding_encoding_format: impl Into<String>,
    ) -> Self {
        self.embedding_encoding_format = Some(embedding_encoding_format.into());
        self
    }

    /// Sets the dimensions for embeddings.
    pub fn embedding_dimensions(mut self, embedding_dimensions: u32) -> Self {
        self.embedding_dimensions = Some(embedding_dimensions);
        self
    }

    /// Sets a validation function to verify LLM responses.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a response string and returns Ok(()) if valid,
    ///         or Err with error message if invalid
    pub fn validator<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        self.validator = Some(Box::new(f));
        self
    }

    /// Sets the number of retry attempts for validation failures.
    ///
    /// # Arguments
    ///
    /// * `attempts` - Maximum number of times to retry generating a valid response
    pub fn validator_attempts(mut self, attempts: usize) -> Self {
        self.validator_attempts = attempts;
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

        #[allow(unused_variables)]
        let provider: Box<dyn LLMProvider> = match backend {
            LLMBackend::OpenAI => {
                #[cfg(not(feature = "openai"))]
                return Err(RllmError::InvalidRequest(
                    "OpenAI feature not enabled".to_string(),
                ));

                #[cfg(feature = "openai")]
                {
                    let key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for OpenAI".to_string())
                    })?;
                    Box::new(crate::backends::openai::OpenAI::new(
                        key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                    ))
                }
            }
            LLMBackend::Anthropic => {
                #[cfg(not(feature = "anthropic"))]
                return Err(RllmError::InvalidRequest(
                    "Anthropic feature not enabled".to_string(),
                ));

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
                    Box::new(anthro)
                }
            }
            LLMBackend::Ollama => {
                #[cfg(not(feature = "ollama"))]
                return Err(RllmError::InvalidRequest(
                    "Ollama feature not enabled".to_string(),
                ));

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
                    Box::new(ollama)
                }
            }
            LLMBackend::DeepSeek => {
                #[cfg(not(feature = "deepseek"))]
                return Err(RllmError::InvalidRequest(
                    "DeepSeek feature not enabled".to_string(),
                ));

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

                    Box::new(deepseek)
                }
            }
            LLMBackend::XAI => {
                #[cfg(not(feature = "xai"))]
                return Err(RllmError::InvalidRequest(
                    "XAI feature not enabled".to_string(),
                ));

                #[cfg(feature = "xai")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for XAI".to_string())
                    })?;

                    let xai = crate::backends::xai::XAI::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                    );
                    Box::new(xai)
                }
            }
            LLMBackend::Phind => {
                #[cfg(not(feature = "phind"))]
                return Err(RllmError::InvalidRequest(
                    "Phind feature not enabled".to_string(),
                ));

                #[cfg(feature = "phind")]
                {
                    let phind = crate::backends::phind::Phind::new(
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                    );
                    Box::new(phind)
                }
            },
            LLMBackend::Google => {
                #[cfg(not(feature = "google"))]
                return Err(RllmError::InvalidRequest(
                    "Google feature not enabled".to_string(),
                ));

                #[cfg(feature = "google")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        RllmError::InvalidRequest("No API key provided for Google".to_string())
                    })?;

                    let google = crate::backends::google::Google::new(
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
                    Box::new(google)
                }
            }
        };

        #[allow(unreachable_code)]
        if let Some(validator) = self.validator {
            Ok(Box::new(crate::validated_llm::ValidatedLLM::new(
                provider,
                validator,
                self.validator_attempts,
            )))
        } else {
            Ok(provider)
        }
    }
}


