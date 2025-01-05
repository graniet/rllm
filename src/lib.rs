//! Rust LLM (RLLM) provides a unified interface for interacting with various LLM providers.
//!
//! This crate abstracts away provider-specific implementation details to offer a consistent API
//! for both chat and completion style interactions with language models.

/// Module containing backend implementations for different LLM providers
pub mod backends;
/// Module for building and configuring LLM providers
pub mod builder;
/// Module for chain of LLM providers
pub mod chain;
/// Module for chat-based interactions with LLMs
pub mod chat;
/// Module for text completion interactions with LLMs
pub mod completion;
/// Module defining error types used throughout the crate
pub mod error;

/// Trait combining chat and completion capabilities that all LLM providers must implement
pub trait LLMProvider: chat::ChatProvider + completion::CompletionProvider {}
