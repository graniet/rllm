//! Module for evaluating and comparing responses from multiple LLM providers.
//!
//! This module provides functionality to run the same prompt through multiple LLMs
//! and score their responses using custom evaluation functions.

use crate::{chat::ChatMessage, error::RllmError, LLMProvider};

/// Type alias for scoring functions that evaluate LLM responses
pub type ScoringFn = dyn Fn(&str) -> f32 + Send + Sync + 'static;

/// Evaluator for comparing responses from multiple LLM providers
pub struct LLMEvaluator {
    /// Collection of LLM providers to evaluate
    llms: Vec<Box<dyn LLMProvider>>,
    /// Optional scoring function to evaluate responses
    scoring_fn: Option<Box<ScoringFn>>,
}

impl LLMEvaluator {
    /// Creates a new evaluator with the given LLM providers
    ///
    /// # Arguments
    /// * `llms` - Vector of LLM providers to evaluate
    pub fn new(llms: Vec<Box<dyn LLMProvider>>) -> Self {
        Self {
            llms,
            scoring_fn: None,
        }
    }

    /// Adds a scoring function to evaluate LLM responses
    ///
    /// # Arguments
    /// * `f` - Function that takes a response string and returns a score
    pub fn scoring<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> f32 + Send + Sync + 'static,
    {
        self.scoring_fn = Some(Box::new(f));
        self
    }

    /// Evaluates chat responses from all providers for the given messages
    ///
    /// # Arguments
    /// * `messages` - Chat messages to send to each provider
    ///
    /// # Returns
    /// Vector of evaluation results containing responses and scores
    pub fn evaluate_chat(&self, messages: &[ChatMessage]) -> Result<Vec<EvalResult>, RllmError> {
        let mut results = Vec::new();
        for llm in &self.llms {
            let response = llm.chat(messages)?;
            let score = if let Some(ref func) = self.scoring_fn {
                (func)(&response)
            } else {
                0.0
            };
            results.push(EvalResult {
                text: response,
                score,
            });
        }
        Ok(results)
    }
}

/// Result of evaluating an LLM response
pub struct EvalResult {
    /// The text response from the LLM
    pub text: String,
    /// Score assigned by the scoring function, if any
    pub score: f32,
}
