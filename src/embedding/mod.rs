use crate::error::RllmError;

pub trait EmbeddingProvider {
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, RllmError>;
}
