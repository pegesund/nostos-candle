//! Tensor creation operations FFI exports.
//!
//! Following nostos-nalgebra conventions:
//! - Function names prefixed with `candle_tensor_`
//! - Handle-based return values
//! - Error codes returned, actual values via out parameters or handles

use crate::commands::{Command, DeviceSpec, DTypeSpec, Response};
use crate::error::{set_last_error, CandleError, ResultCode};
use crate::handles::Handle;
use crate::worker::Worker;

/// Create a tensor filled with zeros.
///
/// # Arguments
/// * `shape` - Pointer to shape array
/// * `shape_len` - Length of shape array
/// * `dtype` - Data type (see DTypeSpec)
/// * `device` - Device (see DeviceSpec)
/// * `out_handle` - Output tensor handle
///
/// # Returns
/// 0 on success, negative error code on failure
#[no_mangle]
pub extern "C" fn candle_tensor_zeros(
    shape: *const usize,
    shape_len: usize,
    dtype: i32,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let dtype_spec = unsafe { std::mem::transmute::<i32, DTypeSpec>(dtype) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Zeros {
        shape: shape_slice.to_vec(),
        dtype: dtype_spec,
        device: device_spec,
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

/// Create a tensor filled with ones.
#[no_mangle]
pub extern "C" fn candle_tensor_ones(
    shape: *const usize,
    shape_len: usize,
    dtype: i32,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let dtype_spec = unsafe { std::mem::transmute::<i32, DTypeSpec>(dtype) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Ones {
        shape: shape_slice.to_vec(),
        dtype: dtype_spec,
        device: device_spec,
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

/// Create a tensor filled with random values from a normal distribution.
#[no_mangle]
pub extern "C" fn candle_tensor_randn(
    shape: *const usize,
    shape_len: usize,
    dtype: i32,
    device: i32,
    mean: f64,
    std: f64,
    out_handle: *mut Handle,
) -> i32 {
    if shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let dtype_spec = unsafe { std::mem::transmute::<i32, DTypeSpec>(dtype) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::Randn {
        shape: shape_slice.to_vec(),
        dtype: dtype_spec,
        device: device_spec,
        mean,
        std,
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

/// Create a tensor from f32 data.
#[no_mangle]
pub extern "C" fn candle_tensor_from_slice_f32(
    data: *const f32,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if data.is_null() || shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::FromSliceF32 {
        data: data_slice.to_vec(),
        shape: shape_slice.to_vec(),
        device: device_spec,
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

/// Create a tensor from f64 data.
#[no_mangle]
pub extern "C" fn candle_tensor_from_slice_f64(
    data: *const f64,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if data.is_null() || shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::FromSliceF64 {
        data: data_slice.to_vec(),
        shape: shape_slice.to_vec(),
        device: device_spec,
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

/// Create a tensor from i64 data.
#[no_mangle]
pub extern "C" fn candle_tensor_from_slice_i64(
    data: *const i64,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
    device: i32,
    out_handle: *mut Handle,
) -> i32 {
    if data.is_null() || shape.is_null() || out_handle.is_null() {
        return ResultCode::NullPointer as i32;
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let device_spec = unsafe { std::mem::transmute::<i32, DeviceSpec>(device) };

    let worker = match Worker::global() {
        Ok(w) => w,
        Err(e) => {
            set_last_error(&e);
            return ResultCode::from(&e) as i32;
        }
    };

    let cmd = Command::FromSliceI64 {
        data: data_slice.to_vec(),
        shape: shape_slice.to_vec(),
        device: device_spec,
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
