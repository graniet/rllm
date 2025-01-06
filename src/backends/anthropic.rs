//! Anthropic API client implementation for chat and completion functionality.
//!
//! This module provides integration with Anthropic's Claude models through their API.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::RllmError,
};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

/// Client for interacting with Anthropic's API.
///
/// Provides methods for chat and completion requests using Anthropic's models.
#[derive(Debug)]
pub struct Anthropic {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub system: String,
    pub stream: bool,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    client: Client,
}

/// Request payload for Anthropic's messages API endpoint.
#[derive(Serialize)]
struct AnthropicCompleteRequest<'a> {
    messages: Vec<AnthropicMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

/// Individual message in an Anthropic chat conversation.
#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Response from Anthropic's messages API endpoint.
#[derive(Deserialize)]
struct AnthropicCompleteResponse {
    content: Vec<AnthropicContent>,
}

/// Content block within an Anthropic API response.
#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

impl Anthropic {
    /// Creates a new Anthropic client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature (defaults to 0.7)
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// * `system` - System prompt (defaults to "You are a helpful assistant.")
    /// * `stream` - Whether to stream responses (defaults to false)
    #[allow(clippy::too_many_arguments)]
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
            model: model.unwrap_or_else(|| "claude-3-sonnet-20240229".to_string()),
            max_tokens: max_tokens.unwrap_or(300),
            temperature: temperature.unwrap_or(0.7),
            system: system.unwrap_or_else(|| "You are a helpful assistant.".to_string()),
            timeout_seconds: timeout_seconds.unwrap_or(30),
            stream: stream.unwrap_or(false),
            top_p,
            top_k,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

impl ChatProvider for Anthropic {
    /// Sends a chat request to Anthropic's API.
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
            return Err(RllmError::AuthError(
                "Missing Anthropic API key".to_string(),
            ));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: Some(&self.system),
            stream: Some(self.stream),
            top_p: self.top_p,
            top_k: self.top_k,
        };

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body)
            .send()?
            .error_for_status()?;

        let json_resp: AnthropicCompleteResponse = resp.json()?;
        if json_resp.content.is_empty() {
            return Err(RllmError::ProviderError(
                "No content returned by Anthropic".to_string(),
            ));
        }

        Ok(json_resp.content[0].text.clone())
    }
}

impl CompletionProvider for Anthropic {
    /// Sends a completion request to Anthropic's API.
    ///
    /// Converts the completion request into a chat message format.
    fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RllmError> {
        let chat_message = ChatMessage {
            role: ChatRole::User,
            content: req.prompt.clone(),
        };
        let answer = self.chat(&[chat_message])?;
        Ok(CompletionResponse { text: answer })
    }
}

impl EmbeddingProvider for Anthropic {
    fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, RllmError> {
        Err(RllmError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}
