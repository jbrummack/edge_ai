#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Executorch(#[from] executorch::Error),
    #[error("Output {0} doesnt exist!")]
    NoOutput(usize),
}
