//! Provides validation capabilities for LLM responses through a wrapper implementation.
//! This module allows adding custom validation logic to any LLM provider.

use crate::chat::{ChatMessage, ChatProvider, ChatRole};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
use crate::error::RllmError;
use crate::{builder::ValidatorFn, LLMProvider};

/// A wrapper around an LLM provider that validates responses before returning them.
///
/// This struct implements validation by:
/// 1. Sending the request to the underlying provider
/// 2. Validating the response using the provided validator function
/// 3. If validation fails, retrying with feedback up to the configured number of attempts
pub struct ValidatedLLM {
    /// The wrapped LLM provider
    inner: Box<dyn LLMProvider>,
    /// Function used to validate responses
    validator: Box<ValidatorFn>,
    /// Maximum number of validation attempts before failing
    attempts: usize,
}

impl ValidatedLLM {
    /// Creates a new ValidatedLLM wrapper.
    ///
    /// # Arguments
    ///
    /// * `inner` - The LLM provider to wrap
    /// * `validator` - Function that validates responses
    /// * `attempts` - Maximum number of validation attempts
    pub fn new(inner: Box<dyn LLMProvider>, validator: Box<ValidatorFn>, attempts: usize) -> Self {
        Self {
            inner,
            validator,
            attempts,
        }
    }
}

impl LLMProvider for ValidatedLLM {}

impl ChatProvider for ValidatedLLM {
    /// Sends a chat request and validates the response.
    ///
    /// If validation fails, retries with feedback to the model about the validation error.
    fn chat(&self, messages: &[ChatMessage]) -> Result<String, RllmError> {
        let mut local_messages = messages.to_vec();
        let mut remaining_attempts = self.attempts;

        loop {
            let response = match self.inner.chat(&local_messages) {
                Ok(resp) => resp,
                Err(e) => return Err(e),
            };

            match (self.validator)(&response) {
                Ok(()) => {
                    return Ok(response);
                }
                Err(err) => {
                    remaining_attempts -= 1;
                    if remaining_attempts == 0 {
                        return Err(RllmError::InvalidRequest(format!(
                            "Validation error after max attempts: {}",
                            err
                        )));
                    }

                    local_messages.push(ChatMessage {
                        role: ChatRole::User,
                        content: format!(
                            "Your previous output was invalid because: {}\n\
                             Please try again and produce a valid response.",
                            err
                        ),
                    });
                }
            }
        }
    }
}

impl CompletionProvider for ValidatedLLM {
    /// Sends a completion request and validates the response.
    ///
    /// If validation fails, retries up to the configured number of attempts.
    fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RllmError> {
        let mut remaining_attempts = self.attempts;

        loop {
            let response = match self.inner.complete(req) {
                Ok(resp) => resp,
                Err(e) => return Err(e),
            };

            match (self.validator)(&response.text) {
                Ok(()) => {
                    return Ok(response);
                }
                Err(err) => {
                    remaining_attempts -= 1;
                    if remaining_attempts == 0 {
                        return Err(RllmError::InvalidRequest(format!(
                            "Validation error after max attempts: {}",
                            err
                        )));
                    }
                }
            }
        }
    }
}

impl EmbeddingProvider for ValidatedLLM {
    /// Passes through embedding requests to the inner provider without validation.
    ///
    /// Embeddings are numerical vectors that don't require validation.
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, RllmError> {
        // Pass through to inner provider since embeddings don't need validation
        self.inner.embed(input)
    }
}
