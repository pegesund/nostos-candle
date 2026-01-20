//! Binary tensor operations FFI exports.

use crate::commands::{Command, Response};
use crate::error::{set_last_error, CandleError, ResultCode};
use crate::handles::Handle;
use crate::worker::Worker;

/// Helper macro for binary operations.
macro_rules! binary_op {
    ($name:ident, $cmd:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(a: Handle, b: Handle, out_handle: *mut Handle) -> i32 {
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

            let cmd = Command::$cmd { a, b };

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

// Element-wise addition: result = a + b
binary_op!(candle_tensor_add, Add);

// Element-wise subtraction: result = a - b
binary_op!(candle_tensor_sub, Sub);

// Element-wise multiplication: result = a * b
binary_op!(candle_tensor_mul, Mul);

// Element-wise division: result = a / b
binary_op!(candle_tensor_div, Div);

// Matrix multiplication: result = a @ b
binary_op!(candle_tensor_matmul, MatMul);
