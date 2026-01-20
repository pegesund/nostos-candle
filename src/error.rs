//! Error types for nostos-candle FFI.

use std::ffi::CString;
use std::os::raw::c_char;
use thiserror::Error;

/// Error types for Candle operations.
#[derive(Error, Debug)]
pub enum CandleError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Invalid handle: {0}")]
    InvalidHandle(u64),

    #[error("Worker not initialized")]
    WorkerNotInitialized,

    #[error("Worker channel closed")]
    ChannelClosed,

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Invalid dtype: {0}")]
    InvalidDType(String),

    #[error("Invalid device: {0}")]
    InvalidDevice(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("Null pointer")]
    NullPointer,

    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("Operation timeout")]
    Timeout,
}

/// Result type alias for Candle operations.
pub type Result<T> = std::result::Result<T, CandleError>;

/// FFI result code.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultCode {
    Ok = 0,
    InvalidHandle = -1,
    WorkerNotInitialized = -2,
    ChannelClosed = -3,
    ShapeMismatch = -4,
    InvalidDType = -5,
    InvalidDevice = -6,
    IoError = -7,
    SafetensorsError = -8,
    NullPointer = -9,
    InvalidUtf8 = -10,
    Timeout = -11,
    CandleError = -100,
}

impl From<&CandleError> for ResultCode {
    fn from(err: &CandleError) -> Self {
        match err {
            CandleError::Candle(_) => ResultCode::CandleError,
            CandleError::InvalidHandle(_) => ResultCode::InvalidHandle,
            CandleError::WorkerNotInitialized => ResultCode::WorkerNotInitialized,
            CandleError::ChannelClosed => ResultCode::ChannelClosed,
            CandleError::ShapeMismatch { .. } => ResultCode::ShapeMismatch,
            CandleError::InvalidDType(_) => ResultCode::InvalidDType,
            CandleError::InvalidDevice(_) => ResultCode::InvalidDevice,
            CandleError::Io(_) => ResultCode::IoError,
            CandleError::Safetensors(_) => ResultCode::SafetensorsError,
            CandleError::NullPointer => ResultCode::NullPointer,
            CandleError::InvalidUtf8(_) => ResultCode::InvalidUtf8,
            CandleError::Timeout => ResultCode::Timeout,
        }
    }
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

/// Set the last error message for FFI retrieval.
pub fn set_last_error(err: &CandleError) {
    let msg = CString::new(err.to_string()).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

/// Get the last error message as a C string pointer.
/// The pointer is valid until the next error is set.
#[no_mangle]
pub extern "C" fn candle_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}
