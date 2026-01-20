//! Candle ML extension for Nostos.
//!
//! Provides tensor operations using the Candle ML framework.

use candle_core::{DType, Device, Tensor};
use nostos_extension::*;
use std::sync::Arc;

declare_extension!("candle", "0.1.0", register);

fn register(reg: &mut ExtRegistry) {
    // Tensor creation
    reg.add("Candle.zeros", tensor_zeros);
    reg.add("Candle.ones", tensor_ones);
    reg.add("Candle.randn", tensor_randn);
    reg.add("Candle.fromList", tensor_from_list);

    // Binary operations
    reg.add("Candle.add", tensor_add);
    reg.add("Candle.sub", tensor_sub);
    reg.add("Candle.mul", tensor_mul);
    reg.add("Candle.div", tensor_div);
    reg.add("Candle.matmul", tensor_matmul);

    // Unary operations
    reg.add("Candle.exp", tensor_exp);
    reg.add("Candle.log", tensor_log);
    reg.add("Candle.relu", tensor_relu);
    reg.add("Candle.gelu", tensor_gelu);
    reg.add("Candle.softmax", tensor_softmax);

    // Shape operations
    reg.add("Candle.reshape", tensor_reshape);
    reg.add("Candle.transpose", tensor_transpose);
    reg.add("Candle.squeeze", tensor_squeeze);
    reg.add("Candle.unsqueeze", tensor_unsqueeze);

    // Reductions
    reg.add("Candle.sum", tensor_sum);
    reg.add("Candle.mean", tensor_mean);
    reg.add("Candle.argmax", tensor_argmax);

    // Tensor info
    reg.add("Candle.shape", tensor_shape);
    reg.add("Candle.toList", tensor_to_list);
    reg.add("Candle.clone", tensor_clone);
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
    let result = a.add(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.sub(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.mul(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_div(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.div(b).map_err(|e| e.to_string())?;
    Ok(tensor_to_value(result))
}

fn tensor_matmul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_tensor(&args[0])?;
    let b = value_to_tensor(&args[1])?;
    let result = a.matmul(b).map_err(|e| e.to_string())?;
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

// ==================== Tensor Info ====================

fn tensor_shape(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let dims: Vec<Value> = t.dims().iter().map(|&d| Value::Int(d as i64)).collect();
    Ok(Value::List(Arc::new(dims)))
}

fn tensor_to_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    let flat = t.flatten_all().map_err(|e| e.to_string())?;
    let data: Vec<f32> = flat.to_vec1().map_err(|e| e.to_string())?;
    let shape = t.dims();

    // Reconstruct nested list from shape
    fn build_nested(data: &[f32], shape: &[usize], offset: &mut usize) -> Value {
        if shape.len() == 1 {
            let values: Vec<Value> = data[*offset..*offset + shape[0]]
                .iter()
                .map(|&f| Value::Float(f as f64))
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
        Ok(Value::Float(data[0] as f64))
    } else {
        let mut offset = 0;
        Ok(build_nested(&data, shape, &mut offset))
    }
}

fn tensor_clone(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let t = value_to_tensor(&args[0])?;
    Ok(tensor_to_value(t.clone()))
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
