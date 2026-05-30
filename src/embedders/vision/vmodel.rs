use std::path::Path;

use executorch::{evalue::IntoEValue, module::Module, tensor::Tensor};
use image::DynamicImage;

use crate::embedders::vision::imtensor::ImageTensor;
type EResult<T> = Result<T, crate::error::Error>;

pub struct VModel<'a, const D: usize> {
    module: Module<'a>,
}
impl<'a, const D: usize> VModel<'a, D> {
    pub fn new(file_path: impl AsRef<Path>) -> EResult<Self> {
        let mut module = Module::new(file_path);
        module.load(None)?;
        Ok(Self { module })
    }
    pub fn embed(&mut self, image: DynamicImage) -> EResult<Vec<f32>> {
        let tensor = ImageTensor::<D, f32>::new(image);
        let t_impl = tensor.tensor_ref()?;
        let t = Tensor::new(&t_impl);
        let outputs = self.module.forward(&[t.into_evalue()])?;
        let embeddings = outputs.get(0).ok_or(crate::error::Error::NoOutput(0))?;
        let vec = embeddings
            .as_tensor()
            .into_typed::<f32>()
            .as_array()
            .to_vec();
        Ok(vec)
    }
}
