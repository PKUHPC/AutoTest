#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

#ifdef HICE_USE_CUDA
#include "hice/device/cuda/context_cuda.h"
#endif

namespace hice {

using fill_kernel_fn_type = void (*)(Tensor &tensor, Scalar fill_value,
                                     size_t begin, size_t end);
HICE_DECLARE_DISPATCHER(fill_kernel_dispatcher, fill_kernel_fn_type);

using rand_uniform_kernel_fn_type = void (*)(Tensor &tensor, Scalar min,
                                             Scalar max);
HICE_DECLARE_DISPATCHER(rand_uniform_kernel_dispatcher,
                        rand_uniform_kernel_fn_type);

using rand_normal_kernel_fn_type = void (*)(Tensor &tensor, Scalar mean,
                                             Scalar stddev);
HICE_DECLARE_DISPATCHER(rand_normal_kernel_dispatcher,
                        rand_normal_kernel_fn_type);

HICE_API Tensor empty(ConstIntArrayRef dims, const TensorOptions &options);

HICE_API Tensor empty_like(const Tensor &input);

HICE_API Tensor zeros(ConstIntArrayRef dims, const TensorOptions &options);

HICE_API Tensor full(ConstIntArrayRef dims, Scalar fill_value,
                     const TensorOptions &options);

HICE_API Tensor rand_uniform(ConstIntArrayRef dims, Scalar min, Scalar max,
                             const TensorOptions &options);

HICE_API Tensor rand_normal(ConstIntArrayRef dims, Scalar mean, Scalar stddev,
                             const TensorOptions &options);

// create a dense tensor by copying data from outside
// @param len: length of values in bytes
HICE_API Tensor create(ConstIntArrayRef dims, void *values, size_t len,
                       const TensorOptions &options);

// create a dense tensor by wrapping data from outside.
// NOTE: This function should be used carefully because it assumed that
// length(values) = sizeof(dims)
HICE_API Tensor wrap(ConstIntArrayRef dims, void *values,
                     const TensorOptions &options, bool copy_ = false);

#ifdef HICE_USE_CUDA
HICE_API void set_stream(cudaStream_t stream);

HICE_API void set_cublas_handle(cublasHandle_t handle);

HICE_API void set_cusparse_handle(cusparseHandle_t handle);

#ifdef HICE_USE_CUDNN
HICE_API void set_cudnn_handle(cudnnHandle_t handle);
#endif
#endif

}  // namespace hice
