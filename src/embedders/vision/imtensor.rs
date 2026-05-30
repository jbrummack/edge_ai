use std::marker::PhantomData;

use executorch::tensor::{DimOrderType, Scalar, SizesType, StridesType, TensorImpl};
use image::{DynamicImage, imageops};
pub trait ImConv: Scalar {
    type Scalar: Scalar;
    fn convert_image_to_planar_buffer<const D: usize>(image: DynamicImage) -> Vec<Self::Scalar>;
}

pub struct ImageTensor<const D: usize, S: Scalar + ImConv> {
    s: PhantomData<S>,
    data: Vec<S>,
}
impl<const D: usize, S: ImConv<Scalar = S>> ImageTensor<D, S> {
    pub fn new(image: DynamicImage) -> Self {
        let data: Vec<S> = S::convert_image_to_planar_buffer::<D>(image);
        Self {
            s: PhantomData,
            data,
        }
    }
    const D32: i32 = D as i32;
    const SIZES: &[SizesType] = &[1, 3, Self::D32, Self::D32];
    const DIM_ORDER: &[DimOrderType] = &[0, 1, 2, 3];
    const STRIDES: &[StridesType] = &[
        3 * Self::D32 * Self::D32,
        Self::D32 * Self::D32,
        Self::D32,
        1,
    ];
    pub fn tensor_ref<'a>(&'a self) -> Result<TensorImpl<'a, S>, crate::error::Error> {
        let tensor_impl =
            TensorImpl::from_slice(Self::SIZES, &self.data, Self::DIM_ORDER, Self::STRIDES)?;
        Ok(tensor_impl)
    }
}
impl ImConv for f32 {
    type Scalar = Self;
    fn convert_image_to_planar_buffer<const D: usize>(image: DynamicImage) -> Vec<Self::Scalar> {
        // Resize to ViT input resolution
        let resized = image
            .resize_exact(D as u32, D as u32, imageops::FilterType::Nearest)
            .to_rgb8();
        let width = D;
        let height = D;
        let channels = 3usize;
        // NCHW layout: 1 x 3 x D x D
        // Tensor shape in memory:
        // [R channel][G channel][B channel]
        let mut tensor = vec![0.0f32; channels * width * height];
        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                let idx = y * width + x;
                // Channel-major (planar) layout
                tensor[idx] = r; // R plane
                tensor[width * height + idx] = g; // G plane
                tensor[2 * width * height + idx] = b; // B plane
            }
        }
        tensor
    }
}
#[cfg(feature = "half")]
impl ImConv for half::f16 {
    type Scalar = Self;

    fn convert_image_to_planar_buffer<const D: usize>(image: DynamicImage) -> Vec<Self::Scalar> {
        // Resize to ViT input resolution
        let resized = image
            .resize_exact(D as u32, D as u32, imageops::FilterType::Nearest)
            .to_rgb8();
        let width = D;
        let height = D;
        let channels = 3usize;
        // NCHW layout: 1 x 3 x D x D
        // Tensor shape in memory:
        // [R channel][G channel][B channel]
        let mut tensor = vec![half::f16::from_f32(0.0); channels * width * height];
        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                let idx = y * width + x;
                // Channel-major (planar) layout
                tensor[idx] = half::f16::from_f32(r); // R plane
                tensor[width * height + idx] = half::f16::from_f32(g); // G plane
                tensor[2 * width * height + idx] = half::f16::from_f32(b); // B plane
            }
        }
        tensor
    }
}

#[cfg(feature = "half")]
impl ImConv for half::bf16 {
    type Scalar = Self;

    fn convert_image_to_planar_buffer<const D: usize>(image: DynamicImage) -> Vec<Self::Scalar> {
        // Resize to ViT input resolution
        let resized = image
            .resize_exact(D as u32, D as u32, imageops::FilterType::Nearest)
            .to_rgb8();
        let width = D;
        let height = D;
        let channels = 3usize;
        // NCHW layout: 1 x 3 x D x D
        // Tensor shape in memory:
        // [R channel][G channel][B channel]
        let mut tensor = vec![half::bf16::from_f32(0.0); channels * width * height];
        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                let idx = y * width + x;
                // Channel-major (planar) layout
                tensor[idx] = half::bf16::from_f32(r); // R plane
                tensor[width * height + idx] = half::bf16::from_f32(g); // G plane
                tensor[2 * width * height + idx] = half::bf16::from_f32(b); // B plane
            }
        }
        tensor
    }
}
