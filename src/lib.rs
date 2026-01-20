//! # nostos-candle
//!
//! Candle ML bindings for Nostos with handle-based FFI design.
//!
//! ## Architecture
//!
//! This crate provides FFI bindings for the Candle ML framework, designed for integration
//! with the Nostos runtime. Key design features:
//!
//! - **Handle-based FFI**: All tensors and models are represented as opaque 64-bit handles
//!   that can be safely passed across the FFI boundary. Memory is managed via explicit
//!   free functions (e.g., `candle_tensor_free`).
//!
//! - **Dedicated Worker Thread**: All GPU operations are performed on a single dedicated
//!   worker thread to ensure proper CUDA/Metal context isolation. This prevents context
//!   conflicts when called from multiple threads.
//!
//! - **Command/Channel Pattern**: Commands are sent to the worker thread via channels,
//!   and responses are returned synchronously. This provides clean error handling and
//!   prevents blocking the GPU thread.
//!
//! ## FFI Conventions (following nostos-nalgebra)
//!
//! - Functions are prefixed with `candle_tensor_` or `candle_model_`
//! - Return values are integers (0 = success, negative = error code)
//! - Output values are passed via out-pointers
//! - Errors can be retrieved via `candle_last_error()`
//!
//! ## Example Usage (from C)
//!
//! ```c
//! // Initialize the worker
//! candle_init();
//!
//! // Create a tensor
//! uint64_t handle;
//! size_t shape[] = {2, 3};
//! candle_tensor_zeros(shape, 2, DTYPE_F32, DEVICE_CPU, &handle);
//!
//! // Use the tensor...
//!
//! // Free the tensor
//! candle_tensor_free(handle);
//!
//! // Shutdown
//! candle_shutdown();
//! ```

pub mod commands;
pub mod error;
pub mod handles;
pub mod model;
pub mod ops;
pub mod worker;

use std::ffi::CStr;
use std::os::raw::c_char;

use commands::{Command, DeviceSpec, DTypeSpec, Response};
use error::{set_last_error, CandleError, ResultCode};
use handles::Handle;
use worker::Worker;

// Re-export FFI functions from modules
pub use error::candle_last_error;
pub use handles::{candle_model_count, candle_model_free, candle_tensor_count, candle_tensor_free};
pub use ops::binary::*;
pub use ops::creation::*;
pub use ops::reduction::*;
pub use ops::shape::*;
pub use ops::unary::*;

// ============================================================================
// Initialization and Lifecycle
// ============================================================================

/// Initialize the Candle worker thread.
///
/// This must be called before any other candle functions.
/// It is safe to call multiple times (subsequent calls are no-ops).
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn candle_init() -> i32 {
    match Worker::init() {
        Ok(()) => ResultCode::Ok as i32,
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
    }
}

/// Shutdown the Candle worker thread.
///
/// This should be called when done using Candle to clean up resources.
/// After calling this, `candle_init()` must be called again before using
/// other functions.
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn candle_shutdown() -> i32 {
    match Worker::global() {
        Ok(w) => match w.shutdown() {
            Ok(()) => ResultCode::Ok as i32,
            Err(e) => {
                set_last_error(&e);
                ResultCode::from(&e) as i32
            }
        },
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
    }
}

// ============================================================================
// Tensor Utilities
// ============================================================================

/// Clone a tensor.
#[no_mangle]
pub extern "C" fn candle_tensor_clone(handle: Handle, out_handle: *mut Handle) -> i32 {
    if out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::Clone { handle }) {
        Ok(Response::TensorHandle(h)) => {
            unsafe { *out_handle = h };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Move a tensor to a different device.
#[no_mangle]
pub extern "C" fn candle_tensor_to_device(
    handle: Handle,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ToDevice {
        handle,
        device: device_spec,
    }) {
        Ok(Response::TensorHandle(h)) => {
            unsafe { *out_handle = h };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Convert a tensor to a different dtype.
#[no_mangle]
pub extern "C" fn candle_tensor_to_dtype(
    handle: Handle,
    dtype: i32,
    out_handle: *mut Handle,
) -> i32 {
    if out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let dtype_spec = unsafe { std::mem::transmute::<i32, DTypeSpec>(dtype) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ToDType {
        handle,
        dtype: dtype_spec,
    }) {
        Ok(Response::TensorHandle(h)) => {
            unsafe { *out_handle = h };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get tensor data as f32 array.
///
/// # Arguments
/// * `handle` - Tensor handle
/// * `out_data` - Pointer to output array (must be pre-allocated)
/// * `out_len` - Pointer to output length
#[no_mangle]
pub extern "C" fn candle_tensor_to_vec_f32(
    handle: Handle,
    out_data: *mut f32,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    if out_data.is_null() || out_len.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ToVecF32 { handle }) {
        Ok(Response::DataF32(data)) => {
            let len = data.len().min(max_len);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_data, len);
                *out_len = len;
            }
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get tensor data as f64 array.
#[no_mangle]
pub extern "C" fn candle_tensor_to_vec_f64(
    handle: Handle,
    out_data: *mut f64,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    if out_data.is_null() || out_len.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ToVecF64 { handle }) {
        Ok(Response::DataF64(data)) => {
            let len = data.len().min(max_len);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_data, len);
                *out_len = len;
            }
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get tensor data as i64 array.
#[no_mangle]
pub extern "C" fn candle_tensor_to_vec_i64(
    handle: Handle,
    out_data: *mut i64,
    max_len: usize,
    out_len: *mut usize,
) -> i32 {
    if out_data.is_null() || out_len.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ToVecI64 { handle }) {
        Ok(Response::DataI64(data)) => {
            let len = data.len().min(max_len);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_data, len);
                *out_len = len;
            }
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get tensor shape.
///
/// # Arguments
/// * `handle` - Tensor handle
/// * `out_shape` - Pointer to output shape array
/// * `max_dims` - Maximum number of dimensions
/// * `out_ndim` - Pointer to output number of dimensions
#[no_mangle]
pub extern "C" fn candle_tensor_shape(
    handle: Handle,
    out_shape: *mut usize,
    max_dims: usize,
    out_ndim: *mut usize,
) -> i32 {
    if out_shape.is_null() || out_ndim.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::Shape { handle }) {
        Ok(Response::Shape(shape)) => {
            let ndim = shape.len().min(max_dims);
            unsafe {
                std::ptr::copy_nonoverlapping(shape.as_ptr(), out_shape, ndim);
                *out_ndim = shape.len();
            }
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get tensor number of dimensions.
#[no_mangle]
pub extern "C" fn candle_tensor_ndim(handle: Handle, out_ndim: *mut usize) -> i32 {
    if out_ndim.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::Shape { handle }) {
        Ok(Response::Shape(shape)) => {
            unsafe { *out_ndim = shape.len() };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get total number of elements in tensor.
#[no_mangle]
pub extern "C" fn candle_tensor_numel(handle: Handle, out_numel: *mut usize) -> i32 {
    if out_numel.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::Shape { handle }) {
        Ok(Response::Shape(shape)) => {
            let numel: usize = shape.iter().product();
            unsafe { *out_numel = numel };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

// ============================================================================
// Model Loading (Safetensors)
// ============================================================================

/// Load a safetensors model from a file.
///
/// # Arguments
/// * `path` - Path to the .safetensors file (null-terminated C string)
/// * `device` - Device to load tensors on (see DeviceSpec)
/// * `out_handle` - Output model handle
#[no_mangle]
pub extern "C" fn candle_model_load(
    path: *const c_char,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if path.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            let err = CandleError::InvalidUtf8(e);
            set_last_error(&err);
            return ResultCode::InvalidUtf8 as i32;
        }
    };

    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::LoadSafetensors {
        path: path_str,
        device: device_spec,
    }) {
        Ok(Response::ModelHandle(h)) => {
            unsafe { *out_handle = h };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get a tensor from a loaded model by name.
///
/// # Arguments
/// * `model` - Model handle
/// * `name` - Tensor name (null-terminated C string)
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_model_get_tensor(
    model: Handle,
    name: *const c_char,
    out_handle: *mut Handle,
) -> i32 {
    if name.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            let err = CandleError::InvalidUtf8(e);
            set_last_error(&err);
            return ResultCode::InvalidUtf8 as i32;
        }
    };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ModelGetTensor {
        model,
        name: name_str,
    }) {
        Ok(Response::TensorHandle(h)) => {
            unsafe { *out_handle = h };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

/// Get the number of tensors in a model.
#[no_mangle]
pub extern "C" fn candle_model_tensor_count(model: Handle, out_count: *mut usize) -> i32 {
    if out_count.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    match worker.send(Command::ModelTensorNames { model }) {
        Ok(Response::Names(names)) => {
            unsafe { *out_count = names.len() };
            ResultCode::Ok as i32
        }
        Ok(Response::Error(msg)) => {
            let err = CandleError::Candle(candle_core::Error::Msg(msg));
            set_last_error(&err);
            ResultCode::CandleError as i32
        }
        Err(e) => {
            set_last_error(&e);
            ResultCode::from(&e) as i32
        }
        _ => ResultCode::CandleError as i32,
    }
}

// ============================================================================
// Constants for FFI
// ============================================================================

/// DType constants for FFI.
pub mod dtype {
    pub const F16: i32 = 0;
    pub const BF16: i32 = 1;
    pub const F32: i32 = 2;
    pub const F64: i32 = 3;
    pub const U8: i32 = 4;
    pub const U32: i32 = 5;
    pub const I64: i32 = 6;
}

/// Device constants for FFI.
pub mod device {
    pub const CPU: i32 = 0;
    pub const CUDA0: i32 = 1;
    pub const CUDA1: i32 = 2;
    pub const CUDA2: i32 = 3;
    pub const CUDA3: i32 = 4;
    pub const METAL: i32 = 10;
}
