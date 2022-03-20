#pragma once

#include "hice/basic/cuda/index_helper.cuh" 

namespace hice{

template <typename TScalarType1, typename TScalarType2, typename TOp>
__global__ void unary_loop_kernel(const TScalarType1 *in, TScalarType2 *out, 
                                  IndexHelper idx_hlpr_tensor,
                                  IndexHelper idx_hlpr_result,
                                  int64_t start, int64_t end, TOp op) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int index = i + start;
  if (index < end) {
    auto offset_in = idx_hlpr_tensor.linear_index_to_offset(index);
    auto offset_out = idx_hlpr_result.linear_index_to_offset(index);
    // printf("index=%d, offset_in=%d, offset_out=%d\n",index, offset_in, offset_out);
    out[offset_out] = op(in[offset_in]);
  }
}

template <typename TScalarType1, typename TScalarType2, typename TOp>
__global__ void unary_loop_kernel_basic(const TScalarType1 *in, TScalarType2 *out, 
                                  int64_t start, int64_t end, TOp op) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int index = i + start;
  if (index < end) {
    out[index] = op(in[index]);
  }
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, typename TOp>
__global__ void binary_loop_kernel(const TScalarType1* in1, 
                                   const TScalarType2* in2,
                                   TScalarType3* out, 
                                   IndexHelper idx_hlpr_in1,
                                   IndexHelper idx_hlpr_in2,
                                   IndexHelper idx_hlpr_out,
                                   int64_t start, int64_t end, TOp op) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int index = i + start;
  if (index >= end) {
    return;
  }
  auto offset_in1 = idx_hlpr_in1.linear_index_to_offset(index);
  auto offset_in2 = idx_hlpr_in2.linear_index_to_offset(index);
  auto offset_out = idx_hlpr_out.linear_index_to_offset(index);
  out[offset_out] = op(in1[offset_in1], in2[offset_in2]);
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, typename TOp>
__global__ void binary_loop_kernel_basic(const TScalarType1* in1, 
                                   const TScalarType2* in2,
                                   TScalarType3* out, 
                                   int64_t start, int64_t end, TOp op) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int index = i + start;
  if (index >= end) {
    return;
  }
  out[index] = op(in1[index], in2[index]);
}

} // namespace hice
