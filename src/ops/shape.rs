//! Shape manipulation operations FFI exports.

use crate::commands::{Command, Response};
use crate::error::{set_last_error, CandleError, ResultCode};
use crate::handles::Handle;
use crate::worker::Worker;

/// Reshape a tensor to a new shape.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `shape` - Pointer to new shape array
/// * `shape_len` - Length of shape array
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_reshape(
    handle: Handle,
    shape: *const usize,
    shape_len: usize,
    out_handle: *mut Handle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Reshape {
        handle,
        shape: shape_slice.to_vec(),
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

/// Transpose two dimensions of a tensor.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim1` - First dimension
/// * `dim2` - Second dimension
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_transpose(
    handle: Handle,
    dim1: usize,
    dim2: usize,
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

    let cmd = Command::Transpose { handle, dim1, dim2 };

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

/// Remove a dimension of size 1.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim` - Dimension to squeeze
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_squeeze(
    handle: Handle,
    dim: usize,
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

    let cmd = Command::Squeeze { handle, dim };

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

/// Add a dimension of size 1.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim` - Position for new dimension
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_unsqueeze(
    handle: Handle,
    dim: usize,
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

    let cmd = Command::Unsqueeze { handle, dim };

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
