//! Command/channel pattern for GPU worker communication.
//!
//! Commands are sent to a dedicated worker thread that owns the GPU context,
//! ensuring all GPU operations happen on a single thread for proper isolation.

use candle_core::{DType, Device};
use crossbeam_channel::Sender;

use crate::error::Result;
use crate::handles::Handle;

/// DType specification for FFI.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeSpec {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    F64 = 3,
    U8 = 4,
    U32 = 5,
    I64 = 6,
}

impl From<DTypeSpec> for DType {
    fn from(spec: DTypeSpec) -> Self {
        match spec {
            DTypeSpec::F16 => DType::F16,
            DTypeSpec::BF16 => DType::BF16,
            DTypeSpec::F32 => DType::F32,
            DTypeSpec::F64 => DType::F64,
            DTypeSpec::U8 => DType::U8,
            DTypeSpec::U32 => DType::U32,
            DTypeSpec::I64 => DType::I64,
        }
    }
}

/// Device specification for FFI.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSpec {
    Cpu = 0,
    Cuda0 = 1,
    Cuda1 = 2,
    Cuda2 = 3,
    Cuda3 = 4,
    Metal = 10,
}

impl DeviceSpec {
    pub fn to_device(self) -> Result<Device> {
        match self {
            DeviceSpec::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda0 => Ok(Device::new_cuda(0)?),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda1 => Ok(Device::new_cuda(1)?),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda2 => Ok(Device::new_cuda(2)?),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda3 => Ok(Device::new_cuda(3)?),
            #[cfg(not(feature = "cuda"))]
            DeviceSpec::Cuda0 | DeviceSpec::Cuda1 | DeviceSpec::Cuda2 | DeviceSpec::Cuda3 => {
                Err(crate::error::CandleError::InvalidDevice(
                    "CUDA not enabled".to_string(),
                ))
            }
            #[cfg(feature = "metal")]
            DeviceSpec::Metal => Ok(Device::new_metal(0)?),
            #[cfg(not(feature = "metal"))]
            DeviceSpec::Metal => Err(crate::error::CandleError::InvalidDevice(
                "Metal not enabled".to_string(),
            )),
        }
    }
}

/// Commands that can be sent to the worker thread.
#[derive(Debug)]
pub enum Command {
    // Lifecycle
    Shutdown,

    // Tensor creation
    Zeros {
        shape: Vec<usize>,
        dtype: DTypeSpec,
        device: DeviceSpec,
    },
    Ones {
        shape: Vec<usize>,
        dtype: DTypeSpec,
        device: DeviceSpec,
    },
    Randn {
        shape: Vec<usize>,
        dtype: DTypeSpec,
        device: DeviceSpec,
        mean: f64,
        std: f64,
    },
    FromSliceF32 {
        data: Vec<f32>,
        shape: Vec<usize>,
        device: DeviceSpec,
    },
    FromSliceF64 {
        data: Vec<f64>,
        shape: Vec<usize>,
        device: DeviceSpec,
    },
    FromSliceI64 {
        data: Vec<i64>,
        shape: Vec<usize>,
        device: DeviceSpec,
    },

    // Binary operations
    Add { a: Handle, b: Handle },
    Sub { a: Handle, b: Handle },
    Mul { a: Handle, b: Handle },
    Div { a: Handle, b: Handle },
    MatMul { a: Handle, b: Handle },

    // Unary operations
    Exp { handle: Handle },
    Log { handle: Handle },
    Relu { handle: Handle },
    Gelu { handle: Handle },
    Softmax { handle: Handle, dim: i64 },

    // Shape operations
    Reshape { handle: Handle, shape: Vec<usize> },
    Transpose { handle: Handle, dim1: usize, dim2: usize },
    Squeeze { handle: Handle, dim: usize },
    Unsqueeze { handle: Handle, dim: usize },

    // Reduction operations
    Sum { handle: Handle, dims: Vec<usize>, keepdim: bool },
    Mean { handle: Handle, dims: Vec<usize>, keepdim: bool },
    Argmax { handle: Handle, dim: usize, keepdim: bool },
    Argmin { handle: Handle, dim: usize, keepdim: bool },

    // Tensor utilities
    Clone { handle: Handle },
    ToDevice { handle: Handle, device: DeviceSpec },
    ToDType { handle: Handle, dtype: DTypeSpec },
    ToVecF32 { handle: Handle },
    ToVecF64 { handle: Handle },
    ToVecI64 { handle: Handle },
    Shape { handle: Handle },
    DType { handle: Handle },
    Device { handle: Handle },

    // Model loading
    LoadSafetensors { path: String, device: DeviceSpec },
    ModelGetTensor { model: Handle, name: String },
    ModelTensorNames { model: Handle },
}

/// Response from the worker thread.
#[derive(Debug)]
pub enum Response {
    /// Successful operation with a new tensor handle.
    TensorHandle(Handle),

    /// Successful operation with a new model handle.
    ModelHandle(Handle),

    /// Successful operation with no return value.
    Ok,

    /// Shape information.
    Shape(Vec<usize>),

    /// DType information (as string).
    DType(String),

    /// Device information (as string).
    Device(String),

    /// Tensor data as f32 vector.
    DataF32(Vec<f32>),

    /// Tensor data as f64 vector.
    DataF64(Vec<f64>),

    /// Tensor data as i64 vector.
    DataI64(Vec<i64>),

    /// List of tensor names.
    Names(Vec<String>),

    /// Error response.
    Error(String),

    /// Shutdown acknowledgment.
    Shutdown,
}

impl Response {
    /// Check if this is an error response.
    pub fn is_error(&self) -> bool {
        matches!(self, Response::Error(_))
    }

    /// Get error message if this is an error response.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Response::Error(msg) => Some(msg),
            _ => None,
        }
    }

    /// Extract tensor handle or return error.
    pub fn into_tensor_handle(self) -> Result<Handle> {
        match self {
            Response::TensorHandle(h) => Ok(h),
            Response::Error(e) => Err(crate::error::CandleError::Candle(
                candle_core::Error::Msg(e),
            )),
            _ => Err(crate::error::CandleError::Candle(
                candle_core::Error::Msg("Unexpected response type".to_string()),
            )),
        }
    }

    /// Extract model handle or return error.
    pub fn into_model_handle(self) -> Result<Handle> {
        match self {
            Response::ModelHandle(h) => Ok(h),
            Response::Error(e) => Err(crate::error::CandleError::Candle(
                candle_core::Error::Msg(e),
            )),
            _ => Err(crate::error::CandleError::Candle(
                candle_core::Error::Msg("Unexpected response type".to_string()),
            )),
        }
    }
}

/// Channel pair for worker communication.
pub struct CommandChannel {
    pub sender: Sender<(Command, Sender<Response>)>,
}

impl CommandChannel {
    /// Create a new command channel.
    pub fn new(sender: Sender<(Command, Sender<Response>)>) -> Self {
        Self { sender }
    }

    /// Send a command and wait for the response.
    pub fn send(&self, cmd: Command) -> Result<Response> {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        self.sender
            .send((cmd, resp_tx))
            .map_err(|_| crate::error::CandleError::ChannelClosed)?;
        resp_rx
            .recv()
            .map_err(|_| crate::error::CandleError::ChannelClosed)
    }

    /// Send a command and wait for the response with a timeout.
    pub fn send_timeout(&self, cmd: Command, timeout: std::time::Duration) -> Result<Response> {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        self.sender
            .send((cmd, resp_tx))
            .map_err(|_| crate::error::CandleError::ChannelClosed)?;
        resp_rx
            .recv_timeout(timeout)
            .map_err(|e| match e {
                crossbeam_channel::RecvTimeoutError::Timeout => crate::error::CandleError::Timeout,
                crossbeam_channel::RecvTimeoutError::Disconnected => {
                    crate::error::CandleError::ChannelClosed
                }
            })
    }
}
