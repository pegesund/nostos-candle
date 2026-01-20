//! Unary tensor operations FFI exports.

use crate::commands::{Command, Response};
use crate::error::{set_last_error, CandleError, ResultCode};
use crate::handles::Handle;
use crate::worker::Worker;

/// Helper macro for simple unary operations.
macro_rules! unary_op {
    ($name:ident, $cmd:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(handle: Handle, out_handle: *mut Handle) -> i32 {
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

            let cmd = Command::$cmd { handle };

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
    };
}

// Exponential: result = exp(x)
unary_op!(candle_tensor_exp, Exp);

// Natural logarithm: result = log(x)
unary_op!(candle_tensor_log, Log);

// ReLU activation: result = max(0, x)
unary_op!(candle_tensor_relu, Relu);

// GELU activation (error function variant)
unary_op!(candle_tensor_gelu, Gelu);

/// Softmax along a dimension.
///
/// # Arguments
/// * `handle` - Input tensor handle
/// * `dim` - Dimension to apply softmax
/// * `out_handle` - Output tensor handle
#[no_mangle]
pub extern "C" fn candle_tensor_softmax(
    handle: Handle,
    dim: i64,
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

    let cmd = Command::Softmax { handle, dim };

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
