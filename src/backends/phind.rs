/// Implementation of the Phind LLM provider.
/// This module provides integration with Phind's language model API.
#[cfg(feature = "phind")]
use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::RllmError,
    LLMProvider,
};
use reqwest::blocking::{Client, Response};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;
use serde_json::{json, Value};

/// Represents a Phind LLM client with configuration options.
pub struct Phind {
    /// The model identifier to use (e.g. "Phind-70B")
    pub model: String,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for controlling randomness (0.0-1.0)
    pub temperature: Option<f32>,
    /// System prompt to prepend to conversations
    pub system: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Whether to stream responses
    pub stream: Option<bool>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Base URL for the Phind API
    pub api_base_url: String,
    /// HTTP client for making requests
    client: Client,
}

impl Phind {
    /// Creates a new Phind client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
            model: model.unwrap_or_else(|| "Phind-70B".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            api_base_url: "https://https.extension.phind.com/agent/".to_string(),
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }

    /// Creates the required headers for API requests.
    fn create_headers() -> Result<HeaderMap, RllmError> {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        headers.insert("User-Agent", HeaderValue::from_static(""));
        headers.insert("Accept", HeaderValue::from_static("*/*"));
        headers.insert("Accept-Encoding", HeaderValue::from_static("Identity"));
        Ok(headers)
    }

    /// Parses a single line from the streaming response.
    fn parse_line(line: &str) -> Option<String> {
        let data = line.strip_prefix("data: ")?;
        let json_value: Value = serde_json::from_str(data).ok()?;

        json_value
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?
            .get("content")?
            .as_str()
            .map(String::from)
    }

    /// Parses the complete streaming response into a single string.
    fn parse_stream_response(response_text: &str) -> String {
        response_text
            .split('\n')
            .filter_map(Self::parse_line)
            .collect()
    }

    /// Interprets the API response and handles any errors.
    fn interpret_response(&self, response: Response) -> Result<String, RllmError> {
        let status = response.status();
        match status {
            StatusCode::OK => {
                let response_text = response.text()?;
                let full_text = Self::parse_stream_response(&response_text);
                if full_text.is_empty() {
                    Err(RllmError::ProviderError(
                        "No completion choice returned.".to_string(),
                    ))
                } else {
                    Ok(full_text)
                }
            }
            _ => {
                let error_text = response.text()?;
                let error_json: Value = serde_json::from_str(&error_text)
                    .unwrap_or_else(|_| json!({"error": {"message": "Unknown error"}}));

                let error_message = error_json
                    .get("error")
                    .and_then(|err| err.get("message"))
                    .and_then(|msg| msg.as_str())
                    .unwrap_or("Unexpected error from Phind")
                    .to_string();

                Err(RllmError::ProviderError(format!(
                    "APIError {}: {}",
                    status, error_message
                )))
            }
        }
    }
}

/// Implementation of chat functionality for Phind.
impl ChatProvider for Phind {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String, RllmError> {
        let mut message_history = vec![];
        for m in messages {
            let role_str = match m.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            message_history.push(json!({
                "content": m.content,
                "role": role_str
            }));
        }

        if let Some(system_prompt) = &self.system {
            message_history.insert(
                0,
                json!({
                    "content": system_prompt,
                    "role": "system"
                }),
            );
        }

        let payload = json!({
            "additional_extension_context": "",
            "allow_magic_buttons": true,
            "is_vscode_extension": true,
            "message_history": message_history,
            "requested_model": self.model,
            "user_input": messages
                .iter()
                .rev()
                .find(|m| m.role == ChatRole::User)
                .map(|m| m.content.clone())
                .unwrap_or_default(),
        });

        let headers = Self::create_headers()?;
        let response = self
            .client
            .post(&self.api_base_url)
            .headers(headers)
            .json(&payload)
            .send()?;

        self.interpret_response(response)
    }
}

/// Implementation of completion functionality for Phind.
impl CompletionProvider for Phind {
    fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, RllmError> {
        let chat_resp = self.chat(&[crate::chat::ChatMessage {
            role: ChatRole::User,
            content: _req.prompt.clone(),
        }])?;

        Ok(CompletionResponse { text: chat_resp })
    }
}

/// Implementation of embedding functionality for Phind.
#[cfg(feature = "phind")]
impl EmbeddingProvider for Phind {
    fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, RllmError> {
        Err(RllmError::ProviderError(
            "Phind does not implement embeddings endpoint yet.".into(),
        ))
    }
}

/// Implementation of the LLMProvider trait for Phind.
impl LLMProvider for Phind {}
