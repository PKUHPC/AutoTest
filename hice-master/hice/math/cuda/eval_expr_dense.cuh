#pragma once

#include "hice/device/cuda/context_cuda.h"
#include "hice/core/tensor.h"
#include "hice/math/cuda/loop_kernel_dense.cuh"
#include "hice/math/cuda/reduce_kernel_dense.cuh"

namespace hice {

template <typename TScalarType1, typename TScalarType2, typename TOp>
static void eval_unary_expr(Expression& expr, TOp op, bool non_blocking = false) {
  const Tensor& tensor = expr.input(0);
  Tensor& result = expr.output(0);
  auto data_ptr_tensor = tensor.data<TScalarType1>();
  auto data_ptr_result = result.mutable_data<TScalarType2>();
  auto ndim_tensor = tensor.ndim();
  auto ndim_result = result.ndim();
  ConstIntArrayRef dims_result = result.dims();
  ConstIntArrayRef dims_tensor = tensor.dims();
  ConstIntArrayRef strides_tensor = expr.strides_input(0);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor(ndim_tensor, dims_tensor.data(), strides_tensor.data());
  IndexHelper idx_hlpr_result(ndim_result, dims_result.data(), strides_result.data());

  int64_t start = 0, end = result.size();
  CUDAContext cuda_ctx(tensor.device());
  auto size = end - start;
  const int block_size = 64;
  const int num_blocks = size / block_size + 1;
  unary_loop_kernel<<<num_blocks, block_size, 0, cuda_ctx.stream()>>>(
      data_ptr_tensor, data_ptr_result, 
      idx_hlpr_tensor, idx_hlpr_result, 
      start, end, op);
  // unary_loop_kernel_basic<<<num_blocks, block_size, 0, cuda_ctx.stream()>>>(
  //     data_ptr_tensor, data_ptr_result, 
  //     start, end, op);
  if (!non_blocking) {
    cuda_ctx.synchronize();
  }
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, typename TOp>
static void eval_binary_expr(Expression& expr, TOp op) {
  const Tensor& tensor1 = expr.input(0);
  const Tensor& tensor2 = expr.input(1);
  Tensor& result = expr.output(0);
  auto data_ptr_tensor1 = tensor1.data<TScalarType1>();
  auto data_ptr_tensor2 = tensor2.data<TScalarType2>();
  auto data_ptr_result = result.mutable_data<TScalarType3>();
  auto ndim_result = result.ndim();
  ConstIntArrayRef dims_result = result.dims();
  ConstIntArrayRef strides_tensor1 = expr.strides_input(0);
  ConstIntArrayRef strides_tensor2 = expr.strides_input(1);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor1(ndim_result, dims_result.data(), strides_tensor1.data());
  IndexHelper idx_hlpr_tensor2(ndim_result, dims_result.data(), strides_tensor2.data());
  IndexHelper idx_hlpr_result(ndim_result, dims_result.data(), strides_result.data());
  int64_t start = 0, end = result.size();

  CUDAContext cuda_ctx(tensor1.device());
  auto size = end - start;
  const int block_size = 64;
  const int num_blocks = size / block_size + 1;
  binary_loop_kernel<<<num_blocks, block_size, 0, cuda_ctx.stream()>>>(
                        data_ptr_tensor1, data_ptr_tensor2, 
                        data_ptr_result, idx_hlpr_tensor1, 
                        idx_hlpr_tensor2, idx_hlpr_result, 
                        start, end, op);
  // binary_loop_kernel_basic<<<num_blocks, block_size, 0, cuda_ctx.stream()>>>(
  //                       data_ptr_tensor1, data_ptr_tensor2, 
  //                       data_ptr_result,  
  //                       start, end, op);
  cuda_ctx.synchronize();
}

template <typename TScalarType, typename TArrayType, typename TOp, typename TDevice>
void reduce_launch_func(const TScalarType* in, TScalarType* out,
                        TArrayType strides_in, TArrayType strides_out,
                        TArrayType dims, int64_t num_items, int64_t dim_start,
                        TScalarType init_value, TScalarType factor,
                        TDevice device_infos, TOp op) {
  // int64_t ndim = dims.size();
  // int64_t num_outputs = 1;
  // for (int64_t i = dim_start; i < dims.size(); i++) {
  //   num_outputs *= dims[i];
  // }
  // int64_t inputs_per_output = num_items / num_outputs;
  // // perpare the input and output calculator
  // auto calc_out = OffsetCalculator(ndim - dim_start, dims.data() + dim_start, strides_out.data() + dim_start);
  // auto calc_in = OffsetCalculator(ndim, dims.data(), strides_in.data());
  // auto calc_in_base = OffsetCalculator(ndim - dim_start, dims.data() + dim_start, strides_in.data() + dim_start);
  // const int block_size = 64;
  // const int num_blocks = num_outputs / block_size + 1;
  // // launch kernel
  // CUDAContext cuda_ctx(device_infos);
  // reduce_kernel<<<num_blocks, block_size>>>(in, out, init_value, factor,
  //                                           num_outputs, inputs_per_output,
  //                                           calc_in, calc_in_base, calc_out, op);
  // cuda_ctx.synchronize();

  int64_t ndim = dims.size();
  int64_t num_outputs = 1;
  for (int64_t i = dim_start; i < dims.size(); i++) {
    num_outputs *= dims[i];
  }
  int64_t inputs_per_output = num_items / num_outputs;

  auto calc_out = OffsetCalculator(ndim - dim_start, dims.data() + dim_start,
                                   strides_out.data() + dim_start);
  auto calc_in = OffsetCalculator(ndim, dims.data(), strides_in.data());
  auto calc_in_base = OffsetCalculator(
      ndim - dim_start, dims.data() + dim_start, strides_in.data() + dim_start);
  // Construct the ReduceConfig struct to config the message for launch kernel
  auto config =
      ReduceConfig(sizeof(TScalarType), num_outputs, inputs_per_output);

  if (strides_in[0] == 1) {
    config.set_block_dimension(dims[0], num_outputs);
  } else {
    config.set_block_dimension(dims[dim_start], inputs_per_output);
  }
  int block_width = config.block_width;
  int block_height = config.block_height;
  if (ndim == 0 || strides_in[0] == 1) {
    config.input_mult[0] = config.split_input(block_width);
  } else {
    config.output_mult[0] = config.split_output(block_width);
  }
  if (config.values_per_thread() >= block_height * 16 ||
      config.values_per_thread() >= 256) {
    config.input_mult[1] = config.split_input(block_height);
  } else {
    config.output_mult[1] = config.split_output(block_height);
  }
  if (config.input_mult[1] != 0 && config.values_per_thread() >= 256 &&
      num_outputs <= 4096) {
    config.ctas_per_output = div_up(config.values_per_thread(), 16);
    if (config.ctas_per_output > 65535) {
      config.ctas_per_output = 65535;
    }
    config.input_mult[2] = config.split_input(config.ctas_per_output);
  }
  int shared_memory = config.shared_memory_size();
  dim3 grid = config.grid();
  dim3 block = config.block();
  auto allocator = get_allocator(kCUDA);
  void* buffer = allocator->allocate_raw(config.global_memory_size());
  void* semaphores = allocator->allocate_raw(config.semaphore_size());

  CUDAContext cuda_ctx(device_infos);
  auto reduce = ReduceOp<TScalarType, TOp>(
      op, config, calc_in, calc_in_base, calc_out, in, out, buffer,
      (int*)semaphores, init_value, factor);
  reduce_kernel<<<grid, block, shared_memory, cuda_ctx.stream()>>>(reduce);
  cuda_ctx.synchronize();
}

template <typename TScalarType, typename TOp>
void eval_reduce_expr(Expression& expr, int64_t reduced_dims_size, TScalarType init_value,
                      TScalarType factor, TOp op) {
  // data preparation
  const Tensor& self = expr.input(0);
  Tensor& result = expr.output(0);
  result.fill(init_value);
  auto data_ptr_in = self.data<TScalarType>();
  auto data_ptr_out = result.mutable_data<TScalarType>();

  int num_items = self.size();
  std::vector<int64_t> strides_in = expr.strides_input(0);
  std::vector<int64_t> strides_out = expr.strides_output(0);
  std::vector<int64_t> perm = expr.reorder_dims(strides_out);

  std::vector<int64_t> permuted_strides_in = expr.permute_dims(strides_in, perm);
  std::vector<int64_t> permuted_strides_out = expr.permute_dims(strides_out, perm);
  std::vector<int64_t> dims = expr.permute_dims(self.shape().dimensions(), perm);

  reduce_launch_func(data_ptr_in, data_ptr_out, permuted_strides_in,
                     permuted_strides_out, dims, num_items, reduced_dims_size,
                     init_value, factor, self.device(), op);
}

} // namespace hice
