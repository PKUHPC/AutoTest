#pragma once

#include "hice/core/sparse_tensor.h"
#include "hice/core/tensor.h"
#include "hice/math/matmul.h"
#include "hice/math/cuda/loop_kernel_dense.cuh"

namespace hice {

namespace {

inline cusparseOperation_t trans_option_from_(MatmulOption option) {
  if (option == kNoTrans) {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  } else if (option == kTrans) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  }
}

// copy int*(cuda) into int64_t*(cuda)
void convert_launch(const int* array32, int64_t* array64, int64_t length, CUDAContext& cuda_ctx){
  int64_t start = 0, end = length;
  auto size = end - start;
  const int block_size = 64;
  const int num_blocks = size / block_size + 1;
  auto op = [=]__device__(int a) -> int64_t { return a; };
  unary_loop_kernel_basic<int, int64_t>
    <<<num_blocks, block_size, 0, cuda_ctx.stream()>>>
    (array32, array64, start, end, op);
  cuda_ctx.synchronize();
}

}  // namespace

template<typename scalar_t>
void print_cuda_array(const scalar_t * cuda_ind, int length, const char* name) {
  scalar_t * cpu_ind = new scalar_t[length];
  cudaMemcpy(cpu_ind, cuda_ind, sizeof(scalar_t) * (length), cudaMemcpyDeviceToHost);
  std::cout<<name<<"=";
  for (int i = 0; i < length; i++) {
    std::cout<<cpu_ind[i]<<", ";
  }
  std::cout<<std::endl;
  delete [] cpu_ind;
}

}  // namespace hice