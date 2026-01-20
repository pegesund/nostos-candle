//! Handle-based FFI design for managing Candle tensors and models.
//!
//! This module provides opaque handles that can be safely passed across the FFI boundary.
//! Handles are 64-bit integers that map to Rust objects stored in thread-safe handle maps.

use candle_core::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{CandleError, Result};
use crate::model::SafeTensorModel;

/// Opaque handle type for FFI.
pub type Handle = u64;

/// Invalid handle sentinel value.
pub const INVALID_HANDLE: Handle = 0;

/// Global counter for generating unique handles.
static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique handle.
fn next_handle() -> Handle {
    HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Thread-safe handle map for storing objects.
pub struct HandleMap<T> {
    map: RwLock<HashMap<Handle, T>>,
}

impl<T> Default for HandleMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HandleMap<T> {
    /// Create a new empty handle map.
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    /// Insert an object and return its handle.
    pub fn insert(&self, value: T) -> Handle {
        let handle = next_handle();
        self.map.write().insert(handle, value);
        handle
    }

    /// Get a reference to an object by handle.
    pub fn get<F, R>(&self, handle: Handle, f: F) -> Result<R>
    where
        F: FnOnce(&T) -> R,
    {
        let map = self.map.read();
        map.get(&handle)
            .map(f)
            .ok_or(CandleError::InvalidHandle(handle))
    }

    /// Get a mutable reference to an object by handle.
    pub fn get_mut<F, R>(&self, handle: Handle, f: F) -> Result<R>
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut map = self.map.write();
        map.get_mut(&handle)
            .map(f)
            .ok_or(CandleError::InvalidHandle(handle))
    }

    /// Remove and return an object by handle.
    pub fn remove(&self, handle: Handle) -> Result<T> {
        self.map
            .write()
            .remove(&handle)
            .ok_or(CandleError::InvalidHandle(handle))
    }

    /// Check if a handle exists.
    pub fn contains(&self, handle: Handle) -> bool {
        self.map.read().contains_key(&handle)
    }

    /// Get the number of stored objects.
    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }

    /// Clear all objects.
    pub fn clear(&self) {
        self.map.write().clear();
    }
}

/// Global handle map for tensors.
pub static TENSOR_HANDLES: once_cell::sync::Lazy<HandleMap<Tensor>> =
    once_cell::sync::Lazy::new(HandleMap::new);

/// Global handle map for loaded safetensor models.
pub static MODEL_HANDLES: once_cell::sync::Lazy<HandleMap<SafeTensorModel>> =
    once_cell::sync::Lazy::new(HandleMap::new);

// ============================================================================
// Tensor Handle FFI Functions
// ============================================================================

/// Create a tensor handle from a Tensor.
pub fn tensor_to_handle(tensor: Tensor) -> Handle {
    TENSOR_HANDLES.insert(tensor)
}

/// Get a tensor from a handle, applying a function.
pub fn with_tensor<F, R>(handle: Handle, f: F) -> Result<R>
where
    F: FnOnce(&Tensor) -> R,
{
    TENSOR_HANDLES.get(handle, f)
}

/// Get two tensors from handles, applying a function.
pub fn with_tensors<F, R>(handle_a: Handle, handle_b: Handle, f: F) -> Result<R>
where
    F: FnOnce(&Tensor, &Tensor) -> R,
{
    let map = TENSOR_HANDLES.map.read();
    let a = map.get(&handle_a).ok_or(CandleError::InvalidHandle(handle_a))?;
    let b = map.get(&handle_b).ok_or(CandleError::InvalidHandle(handle_b))?;
    Ok(f(a, b))
}

/// Clone a tensor from a handle.
pub fn clone_tensor(handle: Handle) -> Result<Tensor> {
    TENSOR_HANDLES.get(handle, |t| t.clone())
}

/// Free a tensor handle.
#[no_mangle]
pub extern "C" fn candle_tensor_free(handle: Handle) -> i32 {
    match TENSOR_HANDLES.remove(handle) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get the number of active tensor handles.
#[no_mangle]
pub extern "C" fn candle_tensor_count() -> u64 {
    TENSOR_HANDLES.len() as u64
}

// ============================================================================
// Model Handle FFI Functions
// ============================================================================

/// Create a model handle from a SafeTensorModel.
pub fn model_to_handle(model: SafeTensorModel) -> Handle {
    MODEL_HANDLES.insert(model)
}

/// Get a model from a handle, applying a function.
pub fn with_model<F, R>(handle: Handle, f: F) -> Result<R>
where
    F: FnOnce(&SafeTensorModel) -> R,
{
    MODEL_HANDLES.get(handle, f)
}

/// Free a model handle.
#[no_mangle]
pub extern "C" fn candle_model_free(handle: Handle) -> i32 {
    match MODEL_HANDLES.remove(handle) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get the number of active model handles.
#[no_mangle]
pub extern "C" fn candle_model_count() -> u64 {
    MODEL_HANDLES.len() as u64
}
