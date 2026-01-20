//! Safetensors model loading functionality.

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{CandleError, Result};

/// A loaded safetensors model containing named tensors.
pub struct SafeTensorModel {
    /// The loaded tensors, keyed by name.
    tensors: HashMap<String, Tensor>,
    /// The device the tensors are loaded on.
    device: Device,
}

/// Convert safetensors dtype to candle dtype.
fn safetensors_dtype_to_candle(dtype: safetensors::Dtype) -> Result<candle_core::DType> {
    match dtype {
        safetensors::Dtype::F16 => Ok(candle_core::DType::F16),
        safetensors::Dtype::BF16 => Ok(candle_core::DType::BF16),
        safetensors::Dtype::F32 => Ok(candle_core::DType::F32),
        safetensors::Dtype::F64 => Ok(candle_core::DType::F64),
        safetensors::Dtype::U8 => Ok(candle_core::DType::U8),
        safetensors::Dtype::U32 => Ok(candle_core::DType::U32),
        safetensors::Dtype::I64 => Ok(candle_core::DType::I64),
        other => Err(CandleError::InvalidDType(format!("{:?}", other))),
    }
}

impl SafeTensorModel {
    /// Load a safetensors file from disk.
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let data = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();
        for (name, view) in safetensors.tensors() {
            let dtype = safetensors_dtype_to_candle(view.dtype())?;
            let tensor = Tensor::from_raw_buffer(
                view.data(),
                dtype,
                view.shape(),
                device,
            )?;
            tensors.insert(name.to_string(), tensor);
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get a tensor by name, cloning it.
    pub fn get_cloned(&self, name: &str) -> Option<Tensor> {
        self.tensors.get(name).cloned()
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get the number of tensors in the model.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the model has no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the device the model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
