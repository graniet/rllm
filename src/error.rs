use std::fmt;

/// Error types that can occur when interacting with LLM providers.
#[derive(Debug)]
pub enum RllmError {
    /// HTTP request/response errors
    HttpError(String),
    /// Authentication and authorization errors
    AuthError(String),
    /// Invalid request parameters or format
    InvalidRequest(String),
    /// Errors returned by the LLM provider
    ProviderError(String),
    /// JSON serialization/deserialization errors
    JsonError(String),
}

impl fmt::Display for RllmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RllmError::HttpError(e) => write!(f, "HTTP Error: {}", e),
            RllmError::AuthError(e) => write!(f, "Auth Error: {}", e),
            RllmError::InvalidRequest(e) => write!(f, "Invalid Request: {}", e),
            RllmError::ProviderError(e) => write!(f, "Provider Error: {}", e),
            RllmError::JsonError(e) => write!(f, "JSON Parse Error: {}", e),
        }
    }
}

impl std::error::Error for RllmError {}

/// Converts reqwest HTTP errors into RllmErrors
impl From<reqwest::Error> for RllmError {
    fn from(err: reqwest::Error) -> Self {
        RllmError::HttpError(err.to_string())
    }
}
