#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

// NOTE:
// - ONLY nchw is supported.

// - Nan numbers are propagated in cudnn.

// - deterministic algorithm is used in cudnn.

// - In avg pooling, The number of padded values will be taken into
//   account when computing the average value.

// - For max pooling, the 'indices' tensor returned MUST NOT be modified.
//   Just get it from forward algorithm and pass it to backward algorithm.
//   Here are some detail informations:
//   - When use cudnn, 'indices' will be a empty tensor. Because cudnn does
//     NOT return indices informations in max pooling.
//   - When use mkl-dnn, 'indices' will be avaliable but NOT reliable. Because
//     mkl-dnn return 'indices' informations ONLY in its 'workspace memory'
//     object, of which data type is unknown, we have to infer it from
//     other informations like size_of_type. Which means bugs might occurs
//     with a different version of mkl-dnn. HICE will check it and throw errors
//     if that happen.

// - For time reason, we ONLY do auto-dims-completion for kernels, padding, stride 
//   in avg_fwd avg_bwd.

namespace hice {

const static int DEFAULT_KERNEL_SIZE = 1;
const static int DEFAULT_STRIDE = 1;
const static int DEFAULT_PADDING = 0;

// infer the expected param from user's input.
inline std::vector<int64_t> infer_params(
  ConstIntArrayRef origin_param,
  int64_t ndim_pooling, 
  int64_t default_value) {
  auto length = origin_param.size();
  if (length == ndim_pooling) {
    return std::vector<int64_t>(origin_param.begin(), origin_param.end());
  } else if (length == 0) {
    return std::vector<int64_t>(ndim_pooling, default_value);
  } else {
    HICE_CHECK(length == 1);
    return std::vector<int64_t>(ndim_pooling, origin_param[0]);
  }
}

inline std::vector<int64_t> compute_out_dims(ConstIntArrayRef input_dims,
                                             ConstIntArrayRef kernel_dims,
                                             ConstIntArrayRef padding,
                                             ConstIntArrayRef stride) {
  std::vector<int64_t> dims_output({/* batch= */input_dims[0], 
                                    /* channel= */input_dims[1]});
  for (int i = 2; i < input_dims.size(); ++i) {
    auto sz_ipt = input_dims[i];
    auto sz_krnl = kernel_dims[i - 2];
    auto sz_strd = stride[i - 2];
    auto sz_pad = padding[i - 2];
    auto sz_otpt = (sz_ipt + 2 * sz_pad - sz_krnl) / sz_strd + 1;
    dims_output.push_back(sz_otpt);
  }
  return dims_output;
}

/*
  Pooling AVG forward.

  parameters:
    - input: {NCHW}
    - kernel: {HW}
    - strides: {HW}
    - paddings: {HW}
*/
using pooling_avg_fwd_kernel_fn_type = void (*)(const Tensor& input,
                                                ConstIntArrayRef kernel,
                                                ConstIntArrayRef stride,
                                                ConstIntArrayRef padding,
                                                Tensor& output);

HICE_DECLARE_DISPATCHER(pooling_avg_fwd_dispatcher,
                        pooling_avg_fwd_kernel_fn_type);

// Forward operators
HICE_API Tensor pooling_avg_fwd(const Tensor& input, ConstIntArrayRef kernel,
                                ConstIntArrayRef stride,
                                ConstIntArrayRef padding);

HICE_API Tensor& pooling_avg_fwd(const Tensor& input, ConstIntArrayRef kernel,
                                ConstIntArrayRef stride, ConstIntArrayRef padding, 
                                Tensor& output);




/*
  Pooling AVG backward.

  parameters:
    - input: {NCHW}, input in forward_pooling
    - output: {NCHW}, output in forward_pooling
    - grad_output: gradient of 'output in forward_pooling'
    - kernel: {HW}
    - strides: {HW}
    - paddings: {HW}
    - grad_input: gradient of 'input in forward_pooling'
*/
using pooling_avg_bwd_kernel_fn_type = void (*)(
    const Tensor& input, const Tensor& output, const Tensor& grad_output,
    ConstIntArrayRef kernel, ConstIntArrayRef stride, ConstIntArrayRef padding,
    Tensor& grad_input);

HICE_DECLARE_DISPATCHER(pooling_avg_bwd_dispatcher,
                        pooling_avg_bwd_kernel_fn_type);

// Backward operators
HICE_API Tensor pooling_avg_bwd(const Tensor& input, const Tensor& output,
                                const Tensor& grad_output,
                                ConstIntArrayRef kernel,
                                ConstIntArrayRef stride,
                                ConstIntArrayRef padding);

HICE_API Tensor& pooling_avg_bwd(const Tensor& input, const Tensor& output,
                                const Tensor& grad_output,
                                ConstIntArrayRef kernel, ConstIntArrayRef
                                stride, ConstIntArrayRef padding, 
                                Tensor& grad_input);




/*
  Pooling MAX forward.

  parameters:
    - input: {NCHW}
    - kernel: {HW}
    - strides: {HW}
    - paddings: {HW}
    - indices: indices of max value for input
  return:
    - tuple<output, indices>
*/
using pooling_max_fwd_kernel_fn_type = void (*)(
    const Tensor& input, ConstIntArrayRef kernel, ConstIntArrayRef stride,
    ConstIntArrayRef padding, Tensor& indices, Tensor& output, bool resizable);

HICE_DECLARE_DISPATCHER(pooling_max_fwd_dispatcher,
                        pooling_max_fwd_kernel_fn_type);

// Forward operators
HICE_API std::tuple<Tensor, Tensor> pooling_max_fwd(const Tensor& input,
                                                    ConstIntArrayRef kernel,
                                                    ConstIntArrayRef stride,
                                                    ConstIntArrayRef padding);




/*
  Pooling MAX backward.

  parameters:
    - input: {NCHW}, input in forward_pooling
    - output: {NCHW}, output in forward_pooling
    - grad_output: gradient of 'output in forward_pooling'
    - kernel: {HW}
    - strides: {HW}
    - paddings: {HW}
    - grad_input: gradient of 'input in forward_pooling'
*/
using pooling_max_bwd_kernel_fn_type = void (*)(
    const Tensor& input, const Tensor& output, const Tensor& grad_output,
    const Tensor& indices, ConstIntArrayRef kernel, ConstIntArrayRef stride,
    ConstIntArrayRef padding, Tensor& grad_input, bool resizable);

HICE_DECLARE_DISPATCHER(pooling_max_bwd_dispatcher,
                        pooling_max_bwd_kernel_fn_type);

// Forward operators
HICE_API Tensor pooling_max_bwd(const Tensor& input, const Tensor& output,
                                const Tensor& grad_output,
                                const Tensor& indices, ConstIntArrayRef kernel,
                                ConstIntArrayRef stride,
                                ConstIntArrayRef padding);

}  // namespace hice
