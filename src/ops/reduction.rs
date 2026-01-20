//! Reduction operations FFI exports.

use crate::commands::{Command, Response};
use crate::error::{set_last_error, CandleError, ResultCode};
use crate::handles::Handle;
use crate::worker::Worker;

/// Sum reduction over specified dimensions.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dims` - Pointer to dimensions array
/// * `dims_len` - Length of dimensions array
/// * `keepdim` - Whether to keep the reduced dimensions
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_sum(
    handle: Handle,
    dims: *const usize,
    dims_len: usize,
    keepdim: bool,
    out_handle: *mut Handle,
) -> i32 {
    if dims.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let dims_slice = unsafe { std::slice::from_raw_parts(dims, dims_len) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Sum {
        handle,
        dims: dims_slice.to_vec(),
        keepdim,
    };

    match worker.send(cmd) {
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

/// Sum over all elements (scalar result).
#[no_mangle]
pub extern "C" fn candle_tensor_sum_all(handle: Handle, out_handle: *mut Handle) -> i32 {
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

    // Sum over all dimensions by passing empty dims
    let cmd = Command::Sum {
        handle,
        dims: vec![],
        keepdim: false,
    };

    match worker.send(cmd) {
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

/// Mean reduction over specified dimensions.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dims` - Pointer to dimensions array
/// * `dims_len` - Length of dimensions array
/// * `keepdim` - Whether to keep the reduced dimensions
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_mean(
    handle: Handle,
    dims: *const usize,
    dims_len: usize,
    keepdim: bool,
    out_handle: *mut Handle,
) -> i32 {
    if dims.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let dims_slice = unsafe { std::slice::from_raw_parts(dims, dims_len) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Mean {
        handle,
        dims: dims_slice.to_vec(),
        keepdim,
    };

    match worker.send(cmd) {
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

/// Mean over all elements (scalar result).
#[no_mangle]
pub extern "C" fn candle_tensor_mean_all(handle: Handle, out_handle: *mut Handle) -> i32 {
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

    let cmd = Command::Mean {
        handle,
        dims: vec![],
        keepdim: false,
    };

    match worker.send(cmd) {
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

/// Argmax along a dimension.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim` - Dimension to reduce
/// * `keepdim` - Whether to keep the reduced dimension
/// * `out_handle` - Output tensor handle (i64 dtype)
#[no_mangle]
pub extern "C" fn candle_tensor_argmax(
    handle: Handle,
    dim: usize,
    keepdim: bool,
    out_handle: *mut Handle,
) -> i32 {
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

    let cmd = Command::Argmax { handle, dim, keepdim };

    match worker.send(cmd) {
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

/// Argmin along a dimension.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim` - Dimension to reduce
/// * `keepdim` - Whether to keep the reduced dimension
/// * `out_handle` - Output tensor handle (i64 dtype)
#[no_mangle]
pub extern "C" fn candle_tensor_argmin(
    handle: Handle,
    dim: usize,
    keepdim: bool,
    out_handle: *mut Handle,
) -> i32 {
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

    let cmd = Command::Argmin { handle, dim, keepdim };

    match worker.send(cmd) {
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
