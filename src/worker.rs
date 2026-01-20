//! Dedicated worker thread for GPU context isolation.
//!
//! All GPU operations are performed on a single worker thread to ensure proper
//! CUDA/Metal context management. Commands are sent via channels and responses
//! are returned asynchronously.

use candle_core::{DType, Device, Tensor};
use crossbeam_channel::{Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::commands::{Command, CommandChannel, DeviceSpec, DTypeSpec, Response};
use crate::error::{CandleError, Result};
use crate::handles::{
    clone_tensor, model_to_handle, tensor_to_handle, with_model, with_tensor, with_tensors,
    Handle,
};
use crate::model::SafeTensorModel;

/// Global worker instance.
static WORKER: once_cell::sync::OnceCell<Worker> = once_cell::sync::OnceCell::new();

/// The GPU worker that processes commands on a dedicated thread.
pub struct Worker {
    /// Channel for sending commands to the worker thread.
    channel: CommandChannel,
    /// Flag to indicate if the worker should shut down.
    shutdown: Arc<AtomicBool>,
    /// Handle to the worker thread.
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Initialize the global worker.
    pub fn init() -> Result<()> {
        WORKER.get_or_try_init(|| Self::new())?;
        Ok(())
    }

    /// Get a reference to the global worker.
    pub fn global() -> Result<&'static Worker> {
        WORKER.get().ok_or(CandleError::WorkerNotInitialized)
    }

    /// Create a new worker with a dedicated thread.
    fn new() -> Result<Self> {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<(Command, Sender<Response>)>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let thread = thread::Builder::new()
            .name("candle-gpu-worker".to_string())
            .spawn(move || {
                Self::worker_loop(cmd_rx, shutdown_clone);
            })
            .map_err(|e| CandleError::Io(e))?;

        Ok(Self {
            channel: CommandChannel::new(cmd_tx),
            shutdown,
            thread: Some(thread),
        })
    }

    /// Main worker loop that processes commands.
    fn worker_loop(rx: Receiver<(Command, Sender<Response>)>, shutdown: Arc<AtomicBool>) {
        log::info!("Candle GPU worker started");

        // Cache devices for reuse
        let cpu_device = Device::Cpu;

        while !shutdown.load(Ordering::Relaxed) {
            match rx.recv() {
                Ok((cmd, resp_tx)) => {
                    let response = Self::handle_command(cmd, &cpu_device);

                    if matches!(response, Response::Shutdown) {
                        let _ = resp_tx.send(response);
                        break;
                    }

                    let _ = resp_tx.send(response);
                }
                Err(_) => {
                    // Channel closed, exit loop
                    break;
                }
            }
        }

        log::info!("Candle GPU worker shut down");
    }

    /// Handle a single command and return a response.
    fn handle_command(cmd: Command, _cpu_device: &Device) -> Response {
        match cmd {
            Command::Shutdown => Response::Shutdown,

            // Tensor creation
            Command::Zeros { shape, dtype, device } => {
                Self::create_zeros(shape, dtype, device)
            }
            Command::Ones { shape, dtype, device } => {
                Self::create_ones(shape, dtype, device)
            }
            Command::Randn { shape, dtype, device, mean, std } => {
                Self::create_randn(shape, dtype, device, mean, std)
            }
            Command::FromSliceF32 { data, shape, device } => {
                Self::from_slice_f32(data, shape, device)
            }
            Command::FromSliceF64 { data, shape, device } => {
                Self::from_slice_f64(data, shape, device)
            }
            Command::FromSliceI64 { data, shape, device } => {
                Self::from_slice_i64(data, shape, device)
            }

            // Binary operations
            Command::Add { a, b } => Self::binary_op(a, b, |x, y| x.add(y)),
            Command::Sub { a, b } => Self::binary_op(a, b, |x, y| x.sub(y)),
            Command::Mul { a, b } => Self::binary_op(a, b, |x, y| x.mul(y)),
            Command::Div { a, b } => Self::binary_op(a, b, |x, y| x.div(y)),
            Command::MatMul { a, b } => Self::binary_op(a, b, |x, y| x.matmul(y)),

            // Unary operations
            Command::Exp { handle } => Self::unary_op(handle, |t| t.exp()),
            Command::Log { handle } => Self::unary_op(handle, |t| t.log()),
            Command::Relu { handle } => Self::unary_op(handle, |t| t.relu()),
            Command::Gelu { handle } => Self::unary_op(handle, |t| t.gelu_erf()),
            Command::Softmax { handle, dim } => {
                Self::unary_op(handle, |t| candle_nn::ops::softmax(t, dim as usize))
            }

            // Shape operations
            Command::Reshape { handle, shape } => {
                Self::unary_op(handle, |t| t.reshape(shape.as_slice()))
            }
            Command::Transpose { handle, dim1, dim2 } => {
                Self::unary_op(handle, |t| t.transpose(dim1, dim2))
            }
            Command::Squeeze { handle, dim } => {
                Self::unary_op(handle, |t| t.squeeze(dim))
            }
            Command::Unsqueeze { handle, dim } => {
                Self::unary_op(handle, |t| t.unsqueeze(dim))
            }

            // Reduction operations
            Command::Sum { handle, dims, keepdim } => {
                Self::reduction_op(handle, &dims, keepdim, |t, d| t.sum(d))
            }
            Command::Mean { handle, dims, keepdim } => {
                Self::reduction_op(handle, &dims, keepdim, |t, d| t.mean(d))
            }
            Command::Argmax { handle, dim, keepdim } => {
                Self::unary_op(handle, |t| t.argmax_keepdim(dim).map(|r| if keepdim { r } else { r.squeeze(dim).unwrap_or(r) }))
            }
            Command::Argmin { handle, dim, keepdim } => {
                Self::unary_op(handle, |t| t.argmin_keepdim(dim).map(|r| if keepdim { r } else { r.squeeze(dim).unwrap_or(r) }))
            }

            // Tensor utilities
            Command::Clone { handle } => {
                match clone_tensor(handle) {
                    Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::ToDevice { handle, device } => {
                match device.to_device() {
                    Ok(dev) => Self::unary_op(handle, |t| t.to_device(&dev)),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::ToDType { handle, dtype } => {
                let dt: DType = dtype.into();
                Self::unary_op(handle, |t| t.to_dtype(dt))
            }
            Command::ToVecF32 { handle } => {
                match with_tensor(handle, |t| t.flatten_all()?.to_vec1::<f32>()) {
                    Ok(Ok(data)) => Response::DataF32(data),
                    Ok(Err(e)) => Response::Error(e.to_string()),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::ToVecF64 { handle } => {
                match with_tensor(handle, |t| t.flatten_all()?.to_vec1::<f64>()) {
                    Ok(Ok(data)) => Response::DataF64(data),
                    Ok(Err(e)) => Response::Error(e.to_string()),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::ToVecI64 { handle } => {
                match with_tensor(handle, |t| t.flatten_all()?.to_vec1::<i64>()) {
                    Ok(Ok(data)) => Response::DataI64(data),
                    Ok(Err(e)) => Response::Error(e.to_string()),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::Shape { handle } => {
                match with_tensor(handle, |t| t.dims().to_vec()) {
                    Ok(shape) => Response::Shape(shape),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::DType { handle } => {
                match with_tensor(handle, |t| format!("{:?}", t.dtype())) {
                    Ok(dtype) => Response::DType(dtype),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
            Command::Device { handle } => {
                match with_tensor(handle, |t| format!("{:?}", t.device())) {
                    Ok(device) => Response::Device(device),
                    Err(e) => Response::Error(e.to_string()),
                }
            }

            // Model loading
            Command::LoadSafetensors { path, device } => {
                Self::load_safetensors(&path, device)
            }
            Command::ModelGetTensor { model, name } => {
                Self::model_get_tensor(model, &name)
            }
            Command::ModelTensorNames { model } => {
                match with_model(model, |m| m.tensor_names()) {
                    Ok(names) => Response::Names(names),
                    Err(e) => Response::Error(e.to_string()),
                }
            }
        }
    }

    // Helper functions for command handling

    fn create_zeros(shape: Vec<usize>, dtype: DTypeSpec, device: DeviceSpec) -> Response {
        let dt: DType = dtype.into();
        match device.to_device() {
            Ok(dev) => match Tensor::zeros(shape.as_slice(), dt, &dev) {
                Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn create_ones(shape: Vec<usize>, dtype: DTypeSpec, device: DeviceSpec) -> Response {
        let dt: DType = dtype.into();
        match device.to_device() {
            Ok(dev) => match Tensor::ones(shape.as_slice(), dt, &dev) {
                Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn create_randn(
        shape: Vec<usize>,
        dtype: DTypeSpec,
        device: DeviceSpec,
        mean: f64,
        std: f64,
    ) -> Response {
        let dt: DType = dtype.into();
        match device.to_device() {
            Ok(dev) => match Tensor::randn(mean, std, shape.as_slice(), &dev) {
                Ok(t) => match t.to_dtype(dt) {
                    Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                    Err(e) => Response::Error(e.to_string()),
                },
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn from_slice_f32(data: Vec<f32>, shape: Vec<usize>, device: DeviceSpec) -> Response {
        match device.to_device() {
            Ok(dev) => match Tensor::from_slice(&data, shape.as_slice(), &dev) {
                Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn from_slice_f64(data: Vec<f64>, shape: Vec<usize>, device: DeviceSpec) -> Response {
        match device.to_device() {
            Ok(dev) => match Tensor::from_slice(&data, shape.as_slice(), &dev) {
                Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn from_slice_i64(data: Vec<i64>, shape: Vec<usize>, device: DeviceSpec) -> Response {
        match device.to_device() {
            Ok(dev) => match Tensor::from_slice(&data, shape.as_slice(), &dev) {
                Ok(t) => Response::TensorHandle(tensor_to_handle(t)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn binary_op<F>(a: Handle, b: Handle, op: F) -> Response
    where
        F: FnOnce(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
    {
        match with_tensors(a, b, |ta, tb| op(ta, tb)) {
            Ok(Ok(t)) => Response::TensorHandle(tensor_to_handle(t)),
            Ok(Err(e)) => Response::Error(e.to_string()),
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn unary_op<F>(handle: Handle, op: F) -> Response
    where
        F: FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    {
        match with_tensor(handle, |t| op(t)) {
            Ok(Ok(t)) => Response::TensorHandle(tensor_to_handle(t)),
            Ok(Err(e)) => Response::Error(e.to_string()),
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn reduction_op<F>(handle: Handle, dims: &[usize], _keepdim: bool, op: F) -> Response
    where
        F: FnOnce(&Tensor, &[usize]) -> candle_core::Result<Tensor>,
    {
        match with_tensor(handle, |t| -> candle_core::Result<Tensor> {
            let result = op(t, dims)?;
            Ok(result)
        }) {
            Ok(Ok(t)) => Response::TensorHandle(tensor_to_handle(t)),
            Ok(Err(e)) => Response::Error(e.to_string()),
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn load_safetensors(path: &str, device: DeviceSpec) -> Response {
        match device.to_device() {
            Ok(dev) => match SafeTensorModel::load(path, &dev) {
                Ok(model) => Response::ModelHandle(model_to_handle(model)),
                Err(e) => Response::Error(e.to_string()),
            },
            Err(e) => Response::Error(e.to_string()),
        }
    }

    fn model_get_tensor(model: Handle, name: &str) -> Response {
        match with_model(model, |m| m.get_cloned(name)) {
            Ok(Some(t)) => Response::TensorHandle(tensor_to_handle(t)),
            Ok(None) => Response::Error(format!("Tensor '{}' not found in model", name)),
            Err(e) => Response::Error(e.to_string()),
        }
    }

    /// Send a command to the worker.
    pub fn send(&self, cmd: Command) -> Result<Response> {
        self.channel.send(cmd)
    }

    /// Send a command with a timeout.
    pub fn send_timeout(&self, cmd: Command, timeout: std::time::Duration) -> Result<Response> {
        self.channel.send_timeout(cmd, timeout)
    }

    /// Shutdown the worker thread.
    pub fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.send(Command::Shutdown);
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Try to send shutdown command, ignoring errors
        let _ = self.channel.send(Command::Shutdown);
        // Wait for thread to finish
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
