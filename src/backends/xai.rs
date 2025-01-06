//! X.AI API client implementation for chat and completion functionality.
//!
//! This module provides integration with X.AI's models through their API.

#[cfg(feature = "xai")]
use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    error::RllmError,
    LLMProvider,
};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

/// Client for interacting with X.AI's API.
///
/// Provides methods for chat and completion requests using X.AI's models.
pub struct XAI {
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

#[derive(Serialize)]
struct XAIChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct XAIChatRequest<'a> {
    model: &'a str,
    messages: Vec<XAIChatMessage<'a>>,
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

#[derive(Deserialize)]
struct XAIChatResponse {
    choices: Vec<XAIChatChoice>,
}

#[derive(Deserialize)]
struct XAIChatChoice {
    message: XAIChatMsg,
}

#[derive(Deserialize)]
struct XAIChatMsg {
    content: String,
}

impl XAI {
    /// Creates a new X.AI client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - X.AI API key
    /// * `model` - Model to use (defaults to "grok-2-latest")
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
            model: model.unwrap_or("grok-2-latest".to_string()),
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

impl ChatProvider for XAI {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String, RllmError> {
        if self.api_key.is_empty() {
            return Err(RllmError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut xai_msgs: Vec<XAIChatMessage> = messages
            .iter()
            .map(|m| XAIChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            xai_msgs.insert(
                0,
                XAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = XAIChatRequest {
            model: &self.model,
            messages: xai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
        };

        let resp = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()?
            .error_for_status()?;

        let json_resp: XAIChatResponse = resp.json()?;
        let first_choice =
            json_resp.choices.into_iter().next().ok_or_else(|| {
                RllmError::ProviderError("No choices returned by X.AI".to_string())
            })?;

        Ok(first_choice.message.content)
    }
}

impl CompletionProvider for XAI {
    fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, RllmError> {
        Ok(CompletionResponse {
            text: "X.AI completion not implemented.".into(),
        })
    }
}

impl LLMProvider for XAI {}
