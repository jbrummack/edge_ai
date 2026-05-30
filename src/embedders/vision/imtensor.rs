use executorch::tensor::{DimOrderType, Scalar, SizesType, StridesType, TensorImpl};
use image::{DynamicImage, imageops};
use std::marker::PhantomData;

pub struct ImageTensor<const D: usize, S: Scalar + ImConv> {
    s: PhantomData<S>,
    data: Vec<S>,
}
impl<const D: usize, S: ImConv<Scalar = S> + Scalar> ImageTensor<D, S> {
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

pub trait ImConv: Sized {
    type Scalar: Scalar;

    /// Helper to convert an f32 value into the target Scalar type
    fn from_f32(val: f32) -> Self::Scalar;

    /// Converts image to planar buffer WITHOUT applying mean and standard deviation
    fn convert_image_to_planar_buffer<const D: usize>(image: DynamicImage) -> Vec<Self::Scalar> {
        // We reuse the normalized logic with identity weights to avoid duplication
        Self::convert_image_to_planar_buffer_normalized::<D>(
            image,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
    }

    /// Converts image to planar buffer and normalizes channels using: (pixel - mean) / std
    fn convert_image_to_planar_buffer_normalized<const D: usize>(
        image: DynamicImage,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Vec<Self::Scalar> {
        // Resize to ViT input resolution
        let resized = image
            .resize_exact(D as u32, D as u32, imageops::FilterType::Nearest)
            .to_rgb8();

        let width = D;
        let height = D;
        let channels = 3usize;

        // Allocate space for NCHW layout: 1 x 3 x D x D
        let mut tensor = Vec::with_capacity(channels * width * height);

        tensor.resize_with(channels * width * height, || Self::from_f32(0.0));

        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x as u32, y as u32);

                let r = ((pixel[0] as f32 / 255.0) - mean[0]) / std[0];
                let g = ((pixel[1] as f32 / 255.0) - mean[1]) / std[1];
                let b = ((pixel[2] as f32 / 255.0) - mean[2]) / std[2];

                let idx = y * width + x;

                // Channel-major (planar) layout mapping
                tensor[idx] = Self::from_f32(r); // R plane
                tensor[width * height + idx] = Self::from_f32(g); // G plane
                tensor[2 * width * height + idx] = Self::from_f32(b); // B plane
            }
        }
        tensor
    }
}

impl ImConv for f32 {
    type Scalar = Self;
    #[inline(always)]
    fn from_f32(val: f32) -> Self::Scalar {
        val
    }
}

#[cfg(feature = "half")]
impl ImConv for half::f16 {
    type Scalar = Self;
    #[inline(always)]
    fn from_f32(val: f32) -> Self::Scalar {
        half::f16::from_f32(val)
    }
}

#[cfg(feature = "half")]
impl ImConv for half::bf16 {
    type Scalar = Self;
    #[inline(always)]
    fn from_f32(val: f32) -> Self::Scalar {
        half::bf16::from_f32(val)
    }
}
