//! Integration tests for nostos-candle FFI bindings.

use std::ptr;

// Import the library
use nostos_candle::*;

/// Test helper to initialize the worker before tests.
fn setup() {
    let result = candle_init();
    assert!(result == 0 || result == -2, "candle_init failed: {}", result);
}

#[test]
fn test_init_and_shutdown() {
    let result = candle_init();
    assert!(result == 0, "candle_init should succeed");

    // Second init should also succeed (no-op)
    let result2 = candle_init();
    assert!(result2 == 0, "second candle_init should succeed");
}

#[test]
fn test_tensor_zeros() {
    setup();

    let shape: [usize; 2] = [2, 3];
    let mut handle: u64 = 0;

    let result = candle_tensor_zeros(
        shape.as_ptr(),
        shape.len(),
        dtype::F32,
        device::CPU,
        &mut handle,
    );

    assert_eq!(result, 0, "candle_tensor_zeros should succeed");
    assert_ne!(handle, 0, "handle should not be zero");

    // Check shape
    let mut out_shape: [usize; 4] = [0; 4];
    let mut ndim: usize = 0;
    let result = candle_tensor_shape(handle, out_shape.as_mut_ptr(), 4, &mut ndim);
    assert_eq!(result, 0, "candle_tensor_shape should succeed");
    assert_eq!(ndim, 2, "tensor should have 2 dimensions");
    assert_eq!(out_shape[0], 2, "first dim should be 2");
    assert_eq!(out_shape[1], 3, "second dim should be 3");

    // Check numel
    let mut numel: usize = 0;
    let result = candle_tensor_numel(handle, &mut numel);
    assert_eq!(result, 0, "candle_tensor_numel should succeed");
    assert_eq!(numel, 6, "tensor should have 6 elements");

    // Free tensor
    let result = candle_tensor_free(handle);
    assert_eq!(result, 0, "candle_tensor_free should succeed");
}

#[test]
fn test_tensor_ones() {
    setup();

    let shape: [usize; 2] = [3, 4];
    let mut handle: u64 = 0;

    let result = candle_tensor_ones(
        shape.as_ptr(),
        shape.len(),
        dtype::F32,
        device::CPU,
        &mut handle,
    );

    assert_eq!(result, 0, "candle_tensor_ones should succeed");

    // Get data and verify all ones
    let mut data: [f32; 12] = [0.0; 12];
    let mut len: usize = 0;
    let result = candle_tensor_to_vec_f32(handle, data.as_mut_ptr(), 12, &mut len);
    assert_eq!(result, 0, "candle_tensor_to_vec_f32 should succeed");
    assert_eq!(len, 12, "should have 12 elements");

    for (i, val) in data.iter().enumerate() {
        assert_eq!(*val, 1.0, "element {} should be 1.0", i);
    }

    candle_tensor_free(handle);
}

#[test]
fn test_tensor_from_slice() {
    setup();

    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: [usize; 2] = [2, 3];
    let mut handle: u64 = 0;

    let result = candle_tensor_from_slice_f32(
        data.as_ptr(),
        data.len(),
        shape.as_ptr(),
        shape.len(),
        device::CPU,
        &mut handle,
    );

    assert_eq!(result, 0, "candle_tensor_from_slice_f32 should succeed");

    // Verify data
    let mut out_data: [f32; 6] = [0.0; 6];
    let mut len: usize = 0;
    let result = candle_tensor_to_vec_f32(handle, out_data.as_mut_ptr(), 6, &mut len);
    assert_eq!(result, 0);
    assert_eq!(len, 6);

    for i in 0..6 {
        assert_eq!(out_data[i], data[i], "element {} mismatch", i);
    }

    candle_tensor_free(handle);
}

#[test]
fn test_binary_ops() {
    setup();

    // Create two tensors
    let data_a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let data_b: [f32; 4] = [2.0, 3.0, 4.0, 5.0];
    let shape: [usize; 1] = [4];

    let mut handle_a: u64 = 0;
    let mut handle_b: u64 = 0;

    candle_tensor_from_slice_f32(data_a.as_ptr(), 4, shape.as_ptr(), 1, device::CPU, &mut handle_a);
    candle_tensor_from_slice_f32(data_b.as_ptr(), 4, shape.as_ptr(), 1, device::CPU, &mut handle_b);

    // Test add
    let mut handle_sum: u64 = 0;
    let result = candle_tensor_add(handle_a, handle_b, &mut handle_sum);
    assert_eq!(result, 0, "candle_tensor_add should succeed");

    let mut sum_data: [f32; 4] = [0.0; 4];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_sum, sum_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(sum_data, [3.0, 5.0, 7.0, 9.0], "add result incorrect");

    // Test sub
    let mut handle_diff: u64 = 0;
    let result = candle_tensor_sub(handle_a, handle_b, &mut handle_diff);
    assert_eq!(result, 0, "candle_tensor_sub should succeed");

    let mut diff_data: [f32; 4] = [0.0; 4];
    candle_tensor_to_vec_f32(handle_diff, diff_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(diff_data, [-1.0, -1.0, -1.0, -1.0], "sub result incorrect");

    // Test mul
    let mut handle_prod: u64 = 0;
    let result = candle_tensor_mul(handle_a, handle_b, &mut handle_prod);
    assert_eq!(result, 0, "candle_tensor_mul should succeed");

    let mut prod_data: [f32; 4] = [0.0; 4];
    candle_tensor_to_vec_f32(handle_prod, prod_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(prod_data, [2.0, 6.0, 12.0, 20.0], "mul result incorrect");

    // Test div
    let mut handle_quot: u64 = 0;
    let result = candle_tensor_div(handle_a, handle_b, &mut handle_quot);
    assert_eq!(result, 0, "candle_tensor_div should succeed");

    let mut quot_data: [f32; 4] = [0.0; 4];
    candle_tensor_to_vec_f32(handle_quot, quot_data.as_mut_ptr(), 4, &mut len);
    assert!((quot_data[0] - 0.5).abs() < 1e-6, "div result incorrect");

    // Cleanup
    candle_tensor_free(handle_a);
    candle_tensor_free(handle_b);
    candle_tensor_free(handle_sum);
    candle_tensor_free(handle_diff);
    candle_tensor_free(handle_prod);
    candle_tensor_free(handle_quot);
}

#[test]
fn test_unary_ops() {
    setup();

    let data: [f32; 4] = [1.0, 2.0, 0.5, -1.0];
    let shape: [usize; 1] = [4];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 4, shape.as_ptr(), 1, device::CPU, &mut handle);

    // Test exp
    let mut handle_exp: u64 = 0;
    let result = candle_tensor_exp(handle, &mut handle_exp);
    assert_eq!(result, 0, "candle_tensor_exp should succeed");

    let mut exp_data: [f32; 4] = [0.0; 4];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_exp, exp_data.as_mut_ptr(), 4, &mut len);
    assert!((exp_data[0] - 1.0_f32.exp()).abs() < 1e-5, "exp(1) incorrect");
    assert!((exp_data[1] - 2.0_f32.exp()).abs() < 1e-5, "exp(2) incorrect");

    // Test relu
    let mut handle_relu: u64 = 0;
    let result = candle_tensor_relu(handle, &mut handle_relu);
    assert_eq!(result, 0, "candle_tensor_relu should succeed");

    let mut relu_data: [f32; 4] = [0.0; 4];
    candle_tensor_to_vec_f32(handle_relu, relu_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(relu_data[0], 1.0, "relu(1) should be 1");
    assert_eq!(relu_data[3], 0.0, "relu(-1) should be 0");

    candle_tensor_free(handle);
    candle_tensor_free(handle_exp);
    candle_tensor_free(handle_relu);
}

#[test]
fn test_matmul() {
    setup();

    // 2x3 matrix
    let data_a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape_a: [usize; 2] = [2, 3];

    // 3x2 matrix
    let data_b: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape_b: [usize; 2] = [3, 2];

    let mut handle_a: u64 = 0;
    let mut handle_b: u64 = 0;

    candle_tensor_from_slice_f32(data_a.as_ptr(), 6, shape_a.as_ptr(), 2, device::CPU, &mut handle_a);
    candle_tensor_from_slice_f32(data_b.as_ptr(), 6, shape_b.as_ptr(), 2, device::CPU, &mut handle_b);

    // Matmul should give 2x2 result
    let mut handle_result: u64 = 0;
    let result = candle_tensor_matmul(handle_a, handle_b, &mut handle_result);
    assert_eq!(result, 0, "candle_tensor_matmul should succeed");

    // Check shape is 2x2
    let mut out_shape: [usize; 4] = [0; 4];
    let mut ndim: usize = 0;
    candle_tensor_shape(handle_result, out_shape.as_mut_ptr(), 4, &mut ndim);
    assert_eq!(ndim, 2);
    assert_eq!(out_shape[0], 2);
    assert_eq!(out_shape[1], 2);

    // Check values: [[22, 28], [49, 64]]
    let mut result_data: [f32; 4] = [0.0; 4];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_result, result_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(result_data[0], 22.0, "matmul[0,0] should be 22");
    assert_eq!(result_data[1], 28.0, "matmul[0,1] should be 28");
    assert_eq!(result_data[2], 49.0, "matmul[1,0] should be 49");
    assert_eq!(result_data[3], 64.0, "matmul[1,1] should be 64");

    candle_tensor_free(handle_a);
    candle_tensor_free(handle_b);
    candle_tensor_free(handle_result);
}

#[test]
fn test_reshape() {
    setup();

    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: [usize; 2] = [2, 3];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 6, shape.as_ptr(), 2, device::CPU, &mut handle);

    // Reshape to 3x2
    let new_shape: [usize; 2] = [3, 2];
    let mut handle_reshaped: u64 = 0;
    let result = candle_tensor_reshape(handle, new_shape.as_ptr(), 2, &mut handle_reshaped);
    assert_eq!(result, 0, "candle_tensor_reshape should succeed");

    // Verify new shape
    let mut out_shape: [usize; 4] = [0; 4];
    let mut ndim: usize = 0;
    candle_tensor_shape(handle_reshaped, out_shape.as_mut_ptr(), 4, &mut ndim);
    assert_eq!(ndim, 2);
    assert_eq!(out_shape[0], 3);
    assert_eq!(out_shape[1], 2);

    candle_tensor_free(handle);
    candle_tensor_free(handle_reshaped);
}

#[test]
fn test_transpose() {
    setup();

    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: [usize; 2] = [2, 3];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 6, shape.as_ptr(), 2, device::CPU, &mut handle);

    // Transpose
    let mut handle_transposed: u64 = 0;
    let result = candle_tensor_transpose(handle, 0, 1, &mut handle_transposed);
    assert_eq!(result, 0, "candle_tensor_transpose should succeed");

    // Verify new shape is 3x2
    let mut out_shape: [usize; 4] = [0; 4];
    let mut ndim: usize = 0;
    candle_tensor_shape(handle_transposed, out_shape.as_mut_ptr(), 4, &mut ndim);
    assert_eq!(ndim, 2);
    assert_eq!(out_shape[0], 3);
    assert_eq!(out_shape[1], 2);

    candle_tensor_free(handle);
    candle_tensor_free(handle_transposed);
}

#[test]
fn test_softmax() {
    setup();

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let shape: [usize; 2] = [2, 2];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 4, shape.as_ptr(), 2, device::CPU, &mut handle);

    // Softmax along last dimension
    let mut handle_softmax: u64 = 0;
    let result = candle_tensor_softmax(handle, 1, &mut handle_softmax);
    assert_eq!(result, 0, "candle_tensor_softmax should succeed");

    // Verify each row sums to 1
    let mut softmax_data: [f32; 4] = [0.0; 4];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_softmax, softmax_data.as_mut_ptr(), 4, &mut len);

    let row1_sum = softmax_data[0] + softmax_data[1];
    let row2_sum = softmax_data[2] + softmax_data[3];
    assert!((row1_sum - 1.0).abs() < 1e-5, "first row should sum to 1");
    assert!((row2_sum - 1.0).abs() < 1e-5, "second row should sum to 1");

    candle_tensor_free(handle);
    candle_tensor_free(handle_softmax);
}

#[test]
fn test_sum_reduction() {
    setup();

    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: [usize; 2] = [2, 3];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 6, shape.as_ptr(), 2, device::CPU, &mut handle);

    // Sum along dim 1
    let dims: [usize; 1] = [1];
    let mut handle_sum: u64 = 0;
    let result = candle_tensor_sum(handle, dims.as_ptr(), 1, false, &mut handle_sum);
    assert_eq!(result, 0, "candle_tensor_sum should succeed");

    let mut sum_data: [f32; 2] = [0.0; 2];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_sum, sum_data.as_mut_ptr(), 2, &mut len);

    // Row sums: [1+2+3=6, 4+5+6=15]
    assert_eq!(sum_data[0], 6.0, "first row sum should be 6");
    assert_eq!(sum_data[1], 15.0, "second row sum should be 15");

    candle_tensor_free(handle);
    candle_tensor_free(handle_sum);
}

#[test]
fn test_clone_tensor() {
    setup();

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let shape: [usize; 1] = [4];
    let mut handle: u64 = 0;

    candle_tensor_from_slice_f32(data.as_ptr(), 4, shape.as_ptr(), 1, device::CPU, &mut handle);

    // Clone
    let mut handle_clone: u64 = 0;
    let result = candle_tensor_clone(handle, &mut handle_clone);
    assert_eq!(result, 0, "candle_tensor_clone should succeed");
    assert_ne!(handle_clone, handle, "clone should have different handle");

    // Verify data is the same
    let mut clone_data: [f32; 4] = [0.0; 4];
    let mut len: usize = 0;
    candle_tensor_to_vec_f32(handle_clone, clone_data.as_mut_ptr(), 4, &mut len);
    assert_eq!(clone_data, data);

    candle_tensor_free(handle);
    candle_tensor_free(handle_clone);
}

#[test]
fn test_tensor_count() {
    setup();

    // Just verify the count function works and returns a reasonable value
    let count = candle_tensor_count();
    // Count should be non-negative (it's u64 so always true, but verifies the call works)
    assert!(count < u64::MAX, "count should be a reasonable value");

    // Create and immediately free a tensor to verify it works
    let shape: [usize; 1] = [4];
    let mut handle: u64 = 0;
    let result = candle_tensor_zeros(shape.as_ptr(), 1, dtype::F32, device::CPU, &mut handle);
    assert_eq!(result, 0, "tensor creation should succeed");

    let result = candle_tensor_free(handle);
    assert_eq!(result, 0, "tensor free should succeed");
}

#[test]
fn test_null_pointer_handling() {
    setup();

    let shape: [usize; 1] = [4];

    // Test null out_handle
    let result = candle_tensor_zeros(shape.as_ptr(), 1, dtype::F32, device::CPU, ptr::null_mut());
    assert!(result < 0, "should fail with null out_handle");

    // Test null shape
    let mut handle: u64 = 0;
    let result = candle_tensor_zeros(ptr::null(), 1, dtype::F32, device::CPU, &mut handle);
    assert!(result < 0, "should fail with null shape");
}

#[test]
fn test_invalid_handle() {
    setup();

    // Try to free non-existent handle
    let result = candle_tensor_free(999999);
    assert!(result < 0, "should fail with invalid handle");
}
