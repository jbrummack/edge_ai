use std::path::Path;

use executorch::{
    evalue::{EValue, IntoEValue},
    module::{Module, ModuleBuilder},
    platform::{LogEntry, PlatformImpl},
    tensor::Tensor,
};
use image::DynamicImage;

use crate::embedders::vision::imtensor::ImageTensor;
type EResult<T> = Result<T, crate::error::Error>;

pub struct VModel<'a, const D: usize> {
    module: Module<'a>,
}

impl<'a, const D: usize> VModel<'a, D> {
    pub fn new(file_path: impl AsRef<Path>) -> EResult<Self> {
        let cml_cache = std::env::var("COREML_CACHE_DIR");
        println!("CML CACHE: {cml_cache:?}");
        //std::env::set_var("COREML_CACHE_DIR", documents_dir);
        //let mut module = ModuleBuilder::new(file_path).event_tracer(dumper).build();
        let mut module = Module::new(file_path);
        module.load(None)?;
        Ok(Self { module })
    }
    pub fn forward_with_output<T>(
        &mut self,
        image: DynamicImage,
        closure: impl FnOnce(Vec<EValue<'_>>) -> T,
    ) -> EResult<T> {
        let tensor = ImageTensor::<D, f32>::new(image);
        let t_impl = tensor.tensor_ref()?;
        let t = Tensor::new(&t_impl);
        let outputs = self.module.forward(&[t.into_evalue()])?;
        let res = closure(outputs);
        Ok(res)
    }
    pub fn embed(&mut self, image: DynamicImage) -> EResult<Vec<f32>> {
        let tensor = ImageTensor::<D, f32>::new(image);
        let t_impl = tensor.tensor_ref()?;
        println!("t_impl");
        let t = Tensor::new(&t_impl);
        println!("fwd");

        let outputs = self.module.forward(&[t.into_evalue()])?;
        println!("embs");
        let embeddings = outputs.get(0).ok_or(crate::error::Error::NoOutput(0))?;
        println!("vec");
        let otensor = embeddings.as_tensor();
        //println!("{:?}", otensor.);
        println!("otensor");
        let typed = otensor.as_typed::<f32>();
        println!("typed");
        let vec = typed.as_array_dyn();
        println!("dim {:?}", vec.dim());
        Ok(vec.iter().cloned().collect())
    }
}
