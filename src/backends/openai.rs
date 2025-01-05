//! OpenAI API client implementation for chat and completion functionality.
//!
//! This module provides integration with OpenAI's GPT models through their API.

#[cfg(feature = "openai")]
use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    error::RllmError,
    LLMProvider,
};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

/// Client for interacting with OpenAI's API.
///
/// Provides methods for chat and completion requests using OpenAI's models.
pub struct OpenAI {
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    client: Client,
}

/// Individual message in an OpenAI chat conversation.
#[derive(Serialize)]
struct OpenAIChatMessage<'a> {
    #[allow(dead_code)]
    role: &'a str,
    content: &'a str,
}

/// Request payload for OpenAI's chat API endpoint.
#[derive(Serialize)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

/// Response from OpenAI's chat API endpoint.
#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChatChoice>,
}

/// Individual choice within an OpenAI chat API response.
#[derive(Deserialize)]
struct OpenAIChatChoice {
    message: OpenAIChatMsg,
}

/// Message content within an OpenAI chat API response.
#[derive(Deserialize)]
struct OpenAIChatMsg {
    #[allow(dead_code)]
    role: String,
    content: String,
}

impl OpenAI {
    /// Creates a new OpenAI client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `model` - Model to use (defaults to "gpt-3.5-turbo")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or("gpt-3.5-turbo".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

impl ChatProvider for OpenAI {
    /// Sends a chat request to OpenAI's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    fn chat(&self, messages: &[ChatMessage]) -> Result<String, RllmError> {
        if self.api_key.is_empty() {
            return Err(RllmError::AuthError("Missing OpenAI API key".to_string()));
        }

        let mut openai_msgs: Vec<OpenAIChatMessage> = messages
            .iter()
            .map(|m| OpenAIChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
        };

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()?
            .error_for_status()?;

        let json_resp: OpenAIChatResponse = resp.json()?;
        let first_choice =
            json_resp.choices.into_iter().next().ok_or_else(|| {
                RllmError::ProviderError("No choices returned by OpenAI".to_string())
            })?;

        Ok(first_choice.message.content)
    }
}

impl CompletionProvider for OpenAI {
    /// Sends a completion request to OpenAI's API.
    ///
    /// Currently not implemented.
    fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, RllmError> {
        Ok(CompletionResponse {
            text: "OpenAI completion not implemented.".into(),
        })
    }
}

impl LLMProvider for OpenAI {}
