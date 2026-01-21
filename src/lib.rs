//! Candle ML extension for Nostos.
//!
//! Provides tensor operations using the Candle ML framework.

use candle_core::{DType, Device, Tensor};
use nostos_extension::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

declare_extension!("candle", "0.1.0", register);

// Type ID for tensor maps (loaded safetensors)
const TENSOR_MAP_TYPE_ID: u64 = 2;

// Type ID for tokenizers
const TOKENIZER_TYPE_ID: u64 = 3;

fn register(reg: &mut ExtRegistry) {
    // === Declare opaque types ===
    reg.add_opaque_type("Tensor");
    reg.add_opaque_type("TensorMap");  // Safetensors loaded model
    reg.add_opaque_type("Tokenizer");

    // === Tensor creation ===
    reg.add_fn("Candle.zeros", "(List[Int]) -> Tensor", tensor_zeros);
    reg.add_fn("Candle.ones", "(List[Int]) -> Tensor", tensor_ones);
    reg.add_fn("Candle.randn", "(List[Int]) -> Tensor", tensor_randn);
    reg.add_fn("Candle.fromList", "(List[Float]) -> Tensor", tensor_from_list);
    reg.add_fn("Candle.fromIntList", "(List[Int]) -> Tensor", tensor_from_int_list);
    reg.add_fn("Candle.arange", "(Int) -> Tensor", tensor_arange);

    // === Binary operations ===
    reg.add_fn("Candle.add", "(Tensor, Tensor) -> Tensor", tensor_add);
    reg.add_fn("Candle.sub", "(Tensor, Tensor) -> Tensor", tensor_sub);
    reg.add_fn("Candle.mul", "(Tensor, Tensor) -> Tensor", tensor_mul);
    reg.add_fn("Candle.div", "(Tensor, Tensor) -> Tensor", tensor_div);
    reg.add_fn("Candle.matmul", "(Tensor, Tensor) -> Tensor", tensor_matmul);
    reg.add_fn("Candle.pow", "(Tensor, Float) -> Tensor", tensor_pow);

    // === Unary operations ===
    reg.add_fn("Candle.exp", "(Tensor) -> Tensor", tensor_exp);
    reg.add_fn("Candle.log", "(Tensor) -> Tensor", tensor_log);
    reg.add_fn("Candle.sqrt", "(Tensor) -> Tensor", tensor_sqrt);
    reg.add_fn("Candle.tanh", "(Tensor) -> Tensor", tensor_tanh);
    reg.add_fn("Candle.relu", "(Tensor) -> Tensor", tensor_relu);
    reg.add_fn("Candle.gelu", "(Tensor) -> Tensor", tensor_gelu);
    reg.add_fn("Candle.softmax", "(Tensor, Int) -> Tensor", tensor_softmax);
    reg.add_fn("Candle.neg", "(Tensor) -> Tensor", tensor_neg);
    reg.add_fn("Candle.cos", "(Tensor) -> Tensor", tensor_cos);
    reg.add_fn("Candle.sin", "(Tensor) -> Tensor", tensor_sin);

    // === Shape operations ===
    reg.add_fn("Candle.reshape", "(Tensor, List[Int]) -> Tensor", tensor_reshape);
    reg.add_fn("Candle.transpose", "(Tensor, Int, Int) -> Tensor", tensor_transpose);
    reg.add_fn("Candle.squeeze", "(Tensor, Int) -> Tensor", tensor_squeeze);
    reg.add_fn("Candle.unsqueeze", "(Tensor, Int) -> Tensor", tensor_unsqueeze);
    reg.add_fn("Candle.cat", "(List[Tensor], Int) -> Tensor", tensor_cat);
    reg.add_fn("Candle.narrow", "(Tensor, Int, Int, Int) -> Tensor", tensor_narrow);
    reg.add_fn("Candle.indexSelect", "(Tensor, Int, Tensor) -> Tensor", tensor_index_select);
    reg.add_fn("Candle.contiguous", "(Tensor) -> Tensor", tensor_contiguous);

    // === Reductions ===
    reg.add_fn("Candle.sum", "(Tensor) -> Tensor", tensor_sum);
    reg.add_fn("Candle.mean", "(Tensor) -> Tensor", tensor_mean);
    reg.add_fn("Candle.argmax", "(Tensor, Int) -> Tensor", tensor_argmax);
    reg.add_fn("Candle.var", "(Tensor, Int) -> Tensor", tensor_var);

    // === Tensor info ===
    reg.add_fn("Candle.shape", "(Tensor) -> List[Int]", tensor_shape);
    reg.add_fn("Candle.toList", "(Tensor) -> List[Float]", tensor_to_list);
    reg.add_fn("Candle.clone", "(Tensor) -> Tensor", tensor_clone);
    reg.add_fn("Candle.dtype", "(Tensor) -> String", tensor_dtype);

    // === Neural network operations ===
    reg.add_fn("Candle.layerNorm", "(Tensor, Tensor, Tensor) -> Tensor", layer_norm);
    reg.add_fn("Candle.embedding", "(Tensor, Tensor) -> Tensor", embedding_lookup);
    reg.add_fn("Candle.linear", "(Tensor, Tensor) -> Tensor", linear);
    reg.add_fn("Candle.dropout", "(Tensor, Float) -> Tensor", dropout);

    // === Model loading ===
    reg.add_fn("Candle.loadSafetensors", "(String) -> TensorMap", load_safetensors);
    reg.add_fn("Candle.getTensor", "(TensorMap, String) -> Tensor", get_tensor_from_map);
    reg.add_fn("Candle.listTensors", "(TensorMap) -> List[String]", list_tensors);

    // === Tokenizer ===
    reg.add_fn("Candle.loadTokenizer", "(String) -> Tokenizer", load_tokenizer);
    reg.add_fn("Candle.encode", "(Tokenizer, String) -> List[Int]", tokenizer_encode);
    reg.add_fn("Candle.decode", "(Tokenizer, List[Int]) -> String", tokenizer_decode);
    reg.add_fn("Candle.vocabSize", "(Tokenizer) -> Int", tokenizer_vocab_size);

    // === Attention utilities ===
    reg.add_fn("Candle.createAttentionMask", "(Tensor, Int) -> Tensor", create_attention_mask);
    reg.add_fn("Candle.applyAttentionMask", "(Tensor, Tensor) -> Tensor", apply_attention_mask);

    // === ModernBERT operations ===
    reg.add_fn("Candle.applyRope", "(Tensor, Tensor, Tensor) -> Tensor", apply_rope);
    reg.add_fn("Candle.geglu", "(Tensor, Tensor) -> Tensor", geglu);
    reg.add_fn("Candle.rmsNorm", "(Tensor, Tensor) -> Tensor", rms_norm);
    reg.add_fn("Candle.localAttentionMask", "(Int, Int) -> Tensor", local_attention_mask);
    reg.add_fn("Candle.cast", "(Tensor, String) -> Tensor", tensor_cast);
    reg.add_fn("Candle.silu", "(Tensor) -> Tensor", tensor_silu);
    reg.add_fn("Candle.ropeFreqs", "(Int, Int, Float) -> List[Tensor]", rope_frequencies);
}

// Type ID for tensors in GcNativeHandle
const TENSOR_TYPE_ID: u64 = 1;

// Cleanup function for GC-managed tensors
fn tensor_cleanup(ptr: usize, type_id: u64) {
    if type_id == TENSOR_TYPE_ID && ptr != 0 {
        unsafe {
            let _ = Box::from_raw(ptr as *mut Tensor);
        }
    }
}

// Helper: Create a Value containing a tensor
fn tensor_to_value(tensor: Tensor) -> Value {
    Value::gc_handle(Box::new(tensor), TENSOR_TYPE_ID, tensor_cleanup)
}

// Helper: Extract tensor from Value
fn value_to_tensor(v: &Value) -> Result<&Tensor, String> {
    let handle = v.as_gc_handle()?;
    if handle.type_id != TENSOR_TYPE_ID {
        return Err(format!("Expected Tensor (type_id={}), got type_id={}", TENSOR_TYPE_ID, handle.type_id));
    }
    if handle.ptr == 0 {
        return Err("Tensor handle is null (non-owning clone)".to_string());
    }
    Ok(unsafe { &*(handle.ptr as *const Tensor) })
}

// Helper: Parse shape from Value (list of ints)
fn value_to_shape(v: &Value) -> Result<Vec<usize>, String> {
    let list = v.as_list()?;
    list.iter()
        .map(|v| v.as_i64().map(|i| i as usize))
        .collect()
}

// ==================== Tensor Creation ====================

/// Create a tensor of zeros: zeros([2, 3]) -> Tensor
fn tensor_zeros(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let shape = value_to_shape(&args[0])?;
    let tensor = Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

/// Create a tensor of ones: ones([2, 3]) -> Tensor
fn tensor_ones(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let shape = value_to_shape(&args[0])?;
    let tensor = Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

/// Create a tensor with random normal values: randn([2, 3]) -> Tensor
fn tensor_randn(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let shape = value_to_shape(&args[0])?;
    let tensor = Tensor::randn(0.0f32, 1.0f32, shape.as_slice(), &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

/// Create a tensor from a nested list: fromList([[1,2],[3,4]]) -> Tensor
fn tensor_from_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let list = args[0].as_list()?;

    // Try to infer shape and flatten data
    let (data, shape) = flatten_nested_list(list)?;

    let tensor = Tensor::from_slice(&data, shape.as_slice(), &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

/// Create a 1D tensor with values from 0 to n-1: arange(5) -> [0, 1, 2, 3, 4]
fn tensor_arange(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as u32;
    let tensor = Tensor::arange(0u32, n, &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

/// Create tensor from integer list (for token IDs): fromIntList([101, 7592]) -> Tensor
/// Also handles nested lists: fromIntList([[101, 102], [103, 104]]) -> Tensor [2, 2]
fn tensor_from_int_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let list = args[0].as_list()?;
    let (data, shape) = flatten_int_list(list)?;

    // Create as i64 tensor (compatible with embedding lookups)
    let tensor = Tensor::from_slice(&data, shape.as_slice(), &Device::Cpu)
        .map_err(|e| e.to_string())?;
    Ok(tensor_to_value(tensor))
}

// Helper to flatten nested integer lists
fn flatten_int_list(list: &[Value]) -> Result<(Vec<i64>, Vec<usize>), String> {
    if list.is_empty() {
        return Ok((vec![], vec![0]));
    }

    match &list[0] {
        Value::Int(_) => {
            // 1D: list of integers
            let data: Vec<i64> = list.iter()
                .map(|v| v.as_i64())
                .collect::<Result<_, _>>()?;
            Ok((data, vec![list.len()]))
        }
        Value::List(_) => {
            // ND: list of lists
            let mut all_data = Vec::new();
            let mut inner_shape = None;

            for item in list {
                let inner = item.as_list()?;
                let (data, shape) = flatten_int_list(inner)?;

                if let Some(ref expected) = inner_shape {
                    if &shape != expected {
                        return Err("Inconsistent tensor shape".to_string());
                    }
                } else {
                    inner_shape = Some(shape);
                }

                all_data.extend(data);
            }

            let mut shape = vec![list.len()];
            if let Some(inner) = inner_shape {
                shape.extend(inner);
            }

            Ok((all_data, shape))
        }
        _ => Err("Expected integer or list of integers".to_string())
    }
}

// Helper to flatten nested lists and infer shape
fn flatten_nested_list(list: &[Value]) -> Result<(Vec<f32>, Vec<usize>), String> {
    if list.is_empty() {
        return Ok((vec![], vec![0]));
    }

    // Check if this is a list of numbers or nested lists
    match &list[0] {
        Value::Float(_) | Value::Int(_) => {
            // 1D: list of numbers
            let data: Vec<f32> = list.iter()
                .map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Result<_, _>>()?;
            Ok((data, vec![list.len()]))
        }
        Value::List(_) => {
            // ND: list of lists
            let mut all_data = Vec::new();
            let mut inner_shape = None;

            for item in list {
                let inner = item.as_list()?;
                let (data, shape) = flatten_nested_list(inner)?;

                // Verify consistent shape
                if let Some(ref expected) = inner_shape {
                    if &shape != expected {
                        return Err("Inconsistent tensor shape".to_string());
                    }
                } else {
                    inner_shape = Some(shape);
                }

                all_data.extend(data);
            }

            let mut shape = vec![list.len()];
            if let Some(inner) = inner_shape {
                shape.extend(inner);
            }

            Ok((all_data, shape))
        }
        _ => Err("Expected number or list".to_string()),
    }
}

// ==================== Binary Operations ====================

fn tensor_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.broadcast_add(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.broadcast_sub(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.broadcast_mul(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_div(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.broadcast_div(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_matmul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.matmul(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_pow(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let exp = args[1].as_f64()? as f64;
    let result = t.powf(exp).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

// ==================== Unary Operations ====================

fn tensor_exp(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.exp().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_log(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.log().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_sqrt(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.sqrt().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_tanh(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.tanh().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_neg(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.neg().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_cos(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.cos().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_sin(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.sin().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_relu(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.relu().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_gelu(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.gelu_erf().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_softmax(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let result = candle_nn::ops::softmax(t, dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

// ==================== Shape Operations ====================

fn tensor_reshape(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let shape = value_to_shape(&args[1])?;
    let result = t.reshape(shape.as_slice()).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_transpose(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim1 = args[1].as_i64()? as usize;
    let dim2 = args[2].as_i64()? as usize;
    let result = t.transpose(dim1, dim2).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_squeeze(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let result = t.squeeze(dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_unsqueeze(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let result = t.unsqueeze(dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Concatenate tensors along a dimension: cat([t1, t2], dim) -> Tensor
fn tensor_cat(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let tensor_list = args[0].as_list()?;
    let dim = args[1].as_i64()? as usize;

    let tensors: Vec<&Tensor> = tensor_list
        .iter()
        .map(|v| value_to_tensor(v))
        .collect::<Result<_, _>>()?;

    let result = Tensor::cat(&tensors, dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Slice a tensor: narrow(t, dim, start, len) -> Tensor
fn tensor_narrow(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let start = args[2].as_i64()? as usize;
    let len = args[3].as_i64()? as usize;
    let result = t.narrow(dim, start, len).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Select indices along a dimension: indexSelect(t, dim, indices) -> Tensor
fn tensor_index_select(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let indices = value_to_tensor(&args[2])?;
    let result = t.index_select(indices, dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Make tensor contiguous in memory
fn tensor_contiguous(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = t.contiguous().map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

// ==================== Reductions ====================

fn tensor_sum(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;

    if args.len() > 1 {
        // Sum over specific dimension
        let dim = args[1].as_i64()? as usize;
        let result = t.sum_keepdim(dim).map_err(|e| e.to_string())?;
        Ok(tensor_to_value(result))
    } else {
        // Sum all elements
        let result = t.sum_all().map_err(|e| e.to_string())?;
        Ok(tensor_to_value(result))
    }
}

fn tensor_mean(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;

    if args.len() > 1 {
        // Mean over specific dimension
        let dim = args[1].as_i64()? as usize;
        let result = t.mean_keepdim(dim).map_err(|e| e.to_string())?;
        Ok(tensor_to_value(result))
    } else {
        // Mean all elements
        let result = t.mean_all().map_err(|e| e.to_string())?;
        Ok(tensor_to_value(result))
    }
}

fn tensor_argmax(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    let result = t.argmax_keepdim(dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Variance along a dimension: var(t, dim) -> Tensor
fn tensor_var(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dim = args[1].as_i64()? as usize;
    // Variance with Bessel's correction (ddof=1)
    let result = t.var_keepdim(dim).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

// ==================== Tensor Info ====================

fn tensor_shape(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dims: Vec<Value> = t.dims().iter().map(|&d| Value::Int(d as i64)).collect();
    Ok(Value::List(Arc::new(dims)))
}

fn tensor_to_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let flat = t.flatten_all().map_err(|e| e.to_string())?;
    let shape = t.dims();

    // Extract data based on dtype
    let data: Vec<f64> = match t.dtype() {
        DType::F32 => {
            let vec: Vec<f32> = flat.to_vec1().map_err(|e| e.to_string())?;
            vec.into_iter().map(|v| v as f64).collect()
        }
        DType::F64 => {
            flat.to_vec1().map_err(|e| e.to_string())?
        }
        DType::U32 => {
            let vec: Vec<u32> = flat.to_vec1().map_err(|e| e.to_string())?;
            vec.into_iter().map(|v| v as f64).collect()
        }
        DType::I64 => {
            let vec: Vec<i64> = flat.to_vec1().map_err(|e| e.to_string())?;
            vec.into_iter().map(|v| v as f64).collect()
        }
        DType::U8 => {
            let vec: Vec<u8> = flat.to_vec1().map_err(|e| e.to_string())?;
            vec.into_iter().map(|v| v as f64).collect()
        }
        dtype => return Err(format!("Unsupported dtype for toList: {:?}", dtype)),
    };

    // Reconstruct nested list from shape
    fn build_nested(data: &[f64], shape: &[usize], offset: &mut usize) -> Value {
        if shape.len() == 1 {
            let values: Vec<Value> = data[*offset..*offset + shape[0]]
                .iter()
                .map(|&f| Value::Float(f))
                .collect();
            *offset += shape[0];
            Value::List(Arc::new(values))
        } else {
            let inner_shape = &shape[1..];
            let values: Vec<Value> = (0..shape[0])
                .map(|_| build_nested(data, inner_shape, offset))
                .collect();
            Value::List(Arc::new(values))
        }
    }

    if shape.is_empty() {
        // Scalar
        Ok(Value::Float(data[0]))
    } else {
        let mut offset = 0;
        Ok(build_nested(&data, shape, &mut offset))
    }
}

fn tensor_clone(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    Ok(tensor_to_value(t.clone()))
}

/// Get the dtype of a tensor as a string
fn tensor_dtype(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dtype_str = match t.dtype() {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::U8 => "u8",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
    };
    Ok(Value::String(Arc::new(dtype_str.to_string())))
}

// ==================== Neural Network Operations ====================

/// Layer normalization: layerNorm(x, gamma, beta, eps) -> Tensor
/// Normalizes the last dimension(s) of x
fn layer_norm(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = value_to_tensor(&args[0])?;
    let gamma = value_to_tensor(&args[1])?;
    let beta = value_to_tensor(&args[2])?;
    let eps = if args.len() > 3 {
        args[3].as_f64()? as f64
    } else {
        1e-5
    };

    // Get the normalized shape (last dimension)
    let last_dim = x.dims().len() - 1;

    // Compute mean and variance over last dimension
    let mean = x.mean_keepdim(last_dim).map_err(|e| e.to_string())?;
    let x_centered = x.broadcast_sub(&mean).map_err(|e| e.to_string())?;
    let variance = x_centered.sqr().map_err(|e| e.to_string())?
        .mean_keepdim(last_dim).map_err(|e| e.to_string())?;

    // Normalize
    let eps_tensor = Tensor::new(&[eps as f32], &Device::Cpu).map_err(|e| e.to_string())?;
    let std = variance.broadcast_add(&eps_tensor).map_err(|e| e.to_string())?
        .sqrt().map_err(|e| e.to_string())?;
    let normalized = x_centered.broadcast_div(&std).map_err(|e| e.to_string())?;

    // Apply scale (gamma) and shift (beta)
    let scaled = normalized.broadcast_mul(gamma).map_err(|e| e.to_string())?;
    let result = scaled.broadcast_add(beta).map_err(|e| e.to_string())?;

    Ok(tensor_to_value(result))
}

/// Embedding lookup: embedding(indices, embeddings) -> Tensor
/// indices: [batch, seq_len] of integers
/// embeddings: [vocab_size, hidden_size]
fn embedding_lookup(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let indices = value_to_tensor(&args[0])?;
    let embeddings = value_to_tensor(&args[1])?;

    // Convert indices to appropriate type if needed
    let indices_u32 = if indices.dtype() != DType::U32 {
        indices.to_dtype(DType::U32).map_err(|e| e.to_string())?
    } else {
        indices.clone()
    };

    let result = embeddings.index_select(&indices_u32.flatten_all().map_err(|e| e.to_string())?, 0)
        .map_err(|e| e.to_string())?;

    // Reshape to [batch, seq_len, hidden_size]
    let mut new_shape: Vec<usize> = indices.dims().to_vec();
    new_shape.push(embeddings.dims()[1]);
    let result = result.reshape(new_shape).map_err(|e| e.to_string())?;

    Ok(tensor_to_value(result))
}

/// Linear layer: linear(x, weight, bias) -> Tensor
/// Computes x @ weight.T + bias
fn linear(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = value_to_tensor(&args[0])?;
    let weight = value_to_tensor(&args[1])?;

    // x: [..., in_features], weight: [out_features, in_features]
    // Result: [..., out_features]
    let weight_t = weight.t().map_err(|e| e.to_string())?;

    // Handle batched matmul: for 3D input [batch, seq, features],
    // we need to use broadcast_matmul
    let result = x.broadcast_matmul(&weight_t).map_err(|e| e.to_string())?;

    // Add bias if provided
    if args.len() > 2 {
        let bias = value_to_tensor(&args[2])?;
        let result = result.broadcast_add(bias).map_err(|e| e.to_string())?;
        Ok(tensor_to_value(result))
    } else {
        Ok(tensor_to_value(result))
    }
}

/// Dropout (inference mode - just returns input): dropout(x, p) -> Tensor
fn dropout(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    // In inference mode, dropout is a no-op
    let x = value_to_tensor(&args[0])?;
    Ok(tensor_to_value(x.clone()))
}

// ==================== Model Loading ====================

// Cleanup function for tensor maps
fn tensor_map_cleanup(ptr: usize, type_id: u64) {
    if type_id == TENSOR_MAP_TYPE_ID && ptr != 0 {
        unsafe {
            let _ = Box::from_raw(ptr as *mut HashMap<String, Tensor>);
        }
    }
}

/// Load safetensors file: loadSafetensors(path) -> TensorMap
fn load_safetensors(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let path = args[0].as_string()?;
    let path_str: &str = &path;

    let tensors = candle_core::safetensors::load(path_str, &Device::Cpu)
        .map_err(|e| format!("Failed to load safetensors: {}", e))?;

    let map: HashMap<String, Tensor> = tensors.into_iter().collect();

    Ok(Value::gc_handle(Box::new(map), TENSOR_MAP_TYPE_ID, tensor_map_cleanup))
}

// Helper to extract tensor map from Value
fn value_to_tensor_map(v: &Value) -> Result<&HashMap<String, Tensor>, String> {
    let handle = v.as_gc_handle()?;
    if handle.type_id != TENSOR_MAP_TYPE_ID {
        return Err(format!("Expected TensorMap (type_id={}), got type_id={}", TENSOR_MAP_TYPE_ID, handle.type_id));
    }
    if handle.ptr == 0 {
        return Err("TensorMap handle is null".to_string());
    }
    Ok(unsafe { &*(handle.ptr as *const HashMap<String, Tensor>) })
}

/// Get a tensor from a tensor map: getTensor(map, name) -> Tensor
fn get_tensor_from_map(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let map = value_to_tensor_map(&args[0])?;
    let name = args[1].as_string()?;

    let name_str: &str = &name;
    let tensor = map.get(name_str)
        .ok_or_else(|| format!("Tensor '{}' not found in model", name))?;

    Ok(tensor_to_value(tensor.clone()))
}

/// List all tensor names in a tensor map: listTensors(map) -> [String]
fn list_tensors(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let map = value_to_tensor_map(&args[0])?;

    let names: Vec<Value> = map.keys()
        .map(|k| Value::String(Arc::new(k.clone())))
        .collect();

    Ok(Value::List(Arc::new(names)))
}

// ==================== Tokenizer ====================

// Cleanup function for tokenizers
fn tokenizer_cleanup(ptr: usize, type_id: u64) {
    if type_id == TOKENIZER_TYPE_ID && ptr != 0 {
        unsafe {
            let _ = Box::from_raw(ptr as *mut Tokenizer);
        }
    }
}

// Helper to extract tokenizer from Value
fn value_to_tokenizer(v: &Value) -> Result<&Tokenizer, String> {
    let handle = v.as_gc_handle()?;
    if handle.type_id != TOKENIZER_TYPE_ID {
        return Err(format!("Expected Tokenizer (type_id={}), got type_id={}", TOKENIZER_TYPE_ID, handle.type_id));
    }
    if handle.ptr == 0 {
        return Err("Tokenizer handle is null".to_string());
    }
    Ok(unsafe { &*(handle.ptr as *const Tokenizer) })
}

/// Load a tokenizer from a JSON file: loadTokenizer(path) -> Tokenizer
fn load_tokenizer(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let path = args[0].as_string()?;
    let path_str: &str = &path;

    let tokenizer = Tokenizer::from_file(path_str)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    Ok(Value::gc_handle(Box::new(tokenizer), TOKENIZER_TYPE_ID, tokenizer_cleanup))
}

/// Encode text to token IDs: encode(tokenizer, text) -> [Int]
fn tokenizer_encode(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let tokenizer = value_to_tokenizer(&args[0])?;
    let text = args[1].as_string()?;
    let text_str: &str = &text;

    let encoding = tokenizer.encode(text_str, true)
        .map_err(|e| format!("Encoding failed: {}", e))?;

    let ids: Vec<Value> = encoding.get_ids()
        .iter()
        .map(|&id| Value::Int(id as i64))
        .collect();

    Ok(Value::List(Arc::new(ids)))
}

/// Decode token IDs back to text: decode(tokenizer, ids) -> String
fn tokenizer_decode(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let tokenizer = value_to_tokenizer(&args[0])?;
    let ids_value = args[1].as_list()?;

    let ids: Vec<u32> = ids_value.iter()
        .map(|v| v.as_i64().map(|i| i as u32))
        .collect::<Result<_, _>>()?;

    let text = tokenizer.decode(&ids, true)
        .map_err(|e| format!("Decoding failed: {}", e))?;

    Ok(Value::String(Arc::new(text)))
}

/// Get vocabulary size: vocabSize(tokenizer) -> Int
fn tokenizer_vocab_size(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let tokenizer = value_to_tokenizer(&args[0])?;
    Ok(Value::Int(tokenizer.get_vocab_size(true) as i64))
}

// ==================== Attention Utilities ====================

/// Create attention mask from token IDs (0 = padding, non-0 = valid)
/// createAttentionMask(tokenIds, padTokenId) -> Tensor [batch, 1, 1, seq]
/// Values: 0.0 for valid tokens, -inf for padding
fn create_attention_mask(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let token_ids = value_to_tensor(&args[0])?;
    let pad_id = args[1].as_i64()? as f32;

    // token_ids: [batch, seq]
    let token_f32 = token_ids.to_dtype(DType::F32).map_err(|e| e.to_string())?;

    // Create mask: 1.0 where token != pad_id, 0.0 where token == pad_id
    let pad_tensor = Tensor::new(&[pad_id], &Device::Cpu).map_err(|e| e.to_string())?;

    // We need to compare and create mask
    // Since Candle doesn't have direct != comparison, we'll use a different approach:
    // mask = (token_ids != pad_id) as float, then convert to attention mask

    // For attention: 0.0 means attend, -inf means don't attend
    // So: if token == pad_id, return -inf, else return 0.0

    let batch_size = token_ids.dims()[0];
    let seq_len = token_ids.dims()[1];

    // Get raw values and create mask
    let flat = token_ids.flatten_all().map_err(|e| e.to_string())?;
    let ids: Vec<f32> = if token_ids.dtype() == DType::F32 {
        flat.to_vec1().map_err(|e| e.to_string())?
    } else {
        let ids_i64: Vec<i64> = flat.to_vec1().map_err(|e| e.to_string())?;
        ids_i64.into_iter().map(|i| i as f32).collect()
    };

    let mask_values: Vec<f32> = ids.iter()
        .map(|&id| if id == pad_id { f32::NEG_INFINITY } else { 0.0 })
        .collect();

    let mask = Tensor::from_slice(&mask_values, &[batch_size, seq_len], &Device::Cpu)
        .map_err(|e| e.to_string())?;

    // Reshape to [batch, 1, 1, seq] for broadcasting with attention scores
    let mask = mask.unsqueeze(1).map_err(|e| e.to_string())?;
    let mask = mask.unsqueeze(1).map_err(|e| e.to_string())?;

    Ok(tensor_to_value(mask))
}

/// Apply attention mask to attention scores
/// applyAttentionMask(scores, mask) -> Tensor
/// scores: [batch, heads, seq, seq]
/// mask: [batch, 1, 1, seq] (broadcasts over heads and query positions)
fn apply_attention_mask(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let scores = value_to_tensor(&args[0])?;
    let mask = value_to_tensor(&args[1])?;

    // Add mask to scores (mask has 0 for valid, -inf for masked)
    let result = scores.broadcast_add(mask).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

// ==================== ModernBERT Operations ====================

/// Apply Rotary Position Embeddings (RoPE)
/// applyRope(x, cos, sin) -> Tensor
/// x: [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
/// cos, sin: [seq, head_dim] precomputed
fn apply_rope(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = value_to_tensor(&args[0])?;
    let cos = value_to_tensor(&args[1])?;
    let sin = value_to_tensor(&args[2])?;

    let dims = x.dims();
    let head_dim = dims[dims.len() - 1];
    let half_dim = head_dim / 2;

    // Split x into two halves: x1, x2
    let x1 = x.narrow(dims.len() - 1, 0, half_dim).map_err(|e| e.to_string())?;
    let x2 = x.narrow(dims.len() - 1, half_dim, half_dim).map_err(|e| e.to_string())?;

    // cos and sin need to be broadcast to match x shape
    // They come as [seq, head_dim], need to split to [seq, half_dim]
    let cos_half = cos.narrow(1, 0, half_dim).map_err(|e| e.to_string())?;
    let sin_half = sin.narrow(1, 0, half_dim).map_err(|e| e.to_string())?;

    // Apply rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
    let rot_x1 = x1.broadcast_mul(&cos_half).map_err(|e| e.to_string())?
        .broadcast_sub(&x2.broadcast_mul(&sin_half).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let rot_x2 = x1.broadcast_mul(&sin_half).map_err(|e| e.to_string())?
        .broadcast_add(&x2.broadcast_mul(&cos_half).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    // Concatenate back
    let result = Tensor::cat(&[&rot_x1, &rot_x2], dims.len() - 1)
        .map_err(|e| e.to_string())?;

    Ok(tensor_to_value(result))
}

/// GeGLU activation: geglu(gate, up) -> GELU(gate) * up
/// Used in ModernBERT MLP: output = GELU(x @ W_gate) * (x @ W_up)
fn geglu(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let gate = value_to_tensor(&args[0])?;
    let up = value_to_tensor(&args[1])?;

    // Apply GELU to gate, then multiply with up
    let gate_act = gate.gelu_erf().map_err(|e| e.to_string())?;
    let result = gate_act.broadcast_mul(up).map_err(|e| e.to_string())?;

    Ok(tensor_to_value(result))
}

/// RMS Normalization (used in some transformer variants)
/// rmsNorm(x, weight, eps) -> Tensor
fn rms_norm(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = value_to_tensor(&args[0])?;
    let weight = value_to_tensor(&args[1])?;
    let eps = if args.len() > 2 {
        args[2].as_f64()? as f32
    } else {
        1e-5
    };

    let last_dim = x.dims().len() - 1;

    // RMS = sqrt(mean(x^2))
    let x_sq = x.sqr().map_err(|e| e.to_string())?;
    let mean_sq = x_sq.mean_keepdim(last_dim).map_err(|e| e.to_string())?;

    let eps_tensor = Tensor::new(&[eps], &Device::Cpu).map_err(|e| e.to_string())?;
    let rms = mean_sq.broadcast_add(&eps_tensor).map_err(|e| e.to_string())?
        .sqrt().map_err(|e| e.to_string())?;

    // Normalize and scale
    let normalized = x.broadcast_div(&rms).map_err(|e| e.to_string())?;
    let result = normalized.broadcast_mul(weight).map_err(|e| e.to_string())?;

    Ok(tensor_to_value(result))
}

/// Create local attention mask (sliding window)
/// localAttentionMask(seqLen, windowSize) -> Tensor [1, 1, seq, seq]
/// Returns 0.0 within window, -inf outside window
fn local_attention_mask(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let seq_len = args[0].as_i64()? as usize;
    let window_size = args[1].as_i64()? as usize;
    let half_window = window_size / 2;

    let mut mask_data = Vec::with_capacity(seq_len * seq_len);

    for i in 0..seq_len {
        for j in 0..seq_len {
            let dist = if i >= j { i - j } else { j - i };
            if dist <= half_window {
                mask_data.push(0.0f32);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }

    let mask = Tensor::from_slice(&mask_data, &[1, 1, seq_len, seq_len], &Device::Cpu)
        .map_err(|e| e.to_string())?;

    Ok(tensor_to_value(mask))
}

/// Cast tensor to different dtype: cast(tensor, "f32") -> Tensor
fn tensor_cast(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dtype_str = args[1].as_string()?;
    let dtype_ref: &str = &dtype_str;

    let dtype = match dtype_ref {
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i64" => DType::I64,
        "u32" => DType::U32,
        "u8" => DType::U8,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => return Err(format!("Unknown dtype: {}", dtype_str)),
    };

    let result = t.to_dtype(dtype).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// SiLU activation (Swish): silu(x) -> x * sigmoid(x)
fn tensor_silu(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let result = candle_nn::ops::silu(t).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

/// Compute RoPE frequencies: ropeFreqs(seqLen, headDim, theta) -> [cos, sin]
/// Returns list of two tensors: cos and sin, each of shape [seqLen, headDim]
fn rope_frequencies(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let seq_len = args[0].as_i64()? as usize;
    let head_dim = args[1].as_i64()? as usize;
    let theta = args[2].as_f64()? as f32;

    let half_dim = head_dim / 2;

    // Compute inverse frequencies: 1 / (theta^(2i/dim)) for i in 0..half_dim
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exp = (2 * i) as f32 / head_dim as f32;
            1.0 / theta.powf(exp)
        })
        .collect();

    // Compute angles for each position: positions * inv_freq
    let mut cos_data = Vec::with_capacity(seq_len * head_dim);
    let mut sin_data = Vec::with_capacity(seq_len * head_dim);

    for pos in 0..seq_len {
        let pos_f = pos as f32;
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
        // Duplicate for second half (interleaved RoPE pattern)
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    let cos = Tensor::from_slice(&cos_data, &[seq_len, head_dim], &Device::Cpu)
        .map_err(|e| e.to_string())?;
    let sin = Tensor::from_slice(&sin_data, &[seq_len, head_dim], &Device::Cpu)
        .map_err(|e| e.to_string())?;

    Ok(Value::list(vec![
        tensor_to_value(cos),
        tensor_to_value(sin),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> ExtContext {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ExtContext::new(rt.handle().clone(), tx, Pid(1))
    }

    #[test]
    fn test_zeros() {
        let ctx = make_ctx();
        let shape = Value::list(vec![Value::Int(2), Value::Int(3)]);
        let result = tensor_zeros(&[shape], &ctx).unwrap();

        let t = value_to_tensor(&result).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
    }

    #[test]
    fn test_from_list() {
        let ctx = make_ctx();
        let data = Value::list(vec![
            Value::list(vec![Value::Float(1.0), Value::Float(2.0)]),
            Value::list(vec![Value::Float(3.0), Value::Float(4.0)]),
        ]);
        let result = tensor_from_list(&[data], &ctx).unwrap();

        let t = value_to_tensor(&result).unwrap();
        assert_eq!(t.dims(), &[2, 2]);
    }

    #[test]
    fn test_add() {
        let ctx = make_ctx();
        let shape = Value::list(vec![Value::Int(2)]);
        let a = tensor_ones(&[shape.clone()], &ctx).unwrap();
        let b = tensor_ones(&[shape], &ctx).unwrap();
        let result = tensor_add(&[a, b], &ctx).unwrap();

        let t = value_to_tensor(&result).unwrap();
        let data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![2.0, 2.0]);
    }

    #[test]
    fn test_matmul() {
        let ctx = make_ctx();
        // 2x3 matrix
        let a_data = Value::list(vec![
            Value::list(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(3.0)]),
            Value::list(vec![Value::Float(4.0), Value::Float(5.0), Value::Float(6.0)]),
        ]);
        // 3x2 matrix
        let b_data = Value::list(vec![
            Value::list(vec![Value::Float(1.0), Value::Float(2.0)]),
            Value::list(vec![Value::Float(3.0), Value::Float(4.0)]),
            Value::list(vec![Value::Float(5.0), Value::Float(6.0)]),
        ]);

        let a = tensor_from_list(&[a_data], &ctx).unwrap();
        let b = tensor_from_list(&[b_data], &ctx).unwrap();
        let result = tensor_matmul(&[a, b], &ctx).unwrap();

        let t = value_to_tensor(&result).unwrap();
        assert_eq!(t.dims(), &[2, 2]);

        let data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
