#pragma once

#include <hice/core/tensor.h>
#include <hice/core/dispatch.h>
#include <hice/util/types.h>

namespace hice {

HICE_API bool pooling_avg_fwd_tvm(const Tensor& input, ConstIntArrayRef kernel,
                                      ConstIntArrayRef stride, ConstIntArrayRef padding, 
                                      Tensor& output);

HICE_API bool pooling_avg_bwd_tvm(const Tensor& input, const Tensor& output,
                                    const Tensor& grad_output,
                                    ConstIntArrayRef kernel, ConstIntArrayRef stride,
                                    ConstIntArrayRef padding, Tensor& grad_input);
  
HICE_API bool conv_fwd_tvm(const Tensor &input, const Tensor &weight,
                              ConstIntArrayRef padding,
                              ConstIntArrayRef stride, ConstIntArrayRef dilation,
                              Tensor &output);

HICE_API bool conv_bwd_input_tvm(const Tensor &input, const Tensor &weight, 
                              const Tensor &grad_output,
                              ConstIntArrayRef padding, ConstIntArrayRef stride,
                              ConstIntArrayRef dilation,
                              Tensor& grad_input);

HICE_API bool conv_bwd_weight_tvm(const Tensor &input, const Tensor &weight, 
                              const Tensor &grad_output,
                              ConstIntArrayRef padding, ConstIntArrayRef stride,
                              ConstIntArrayRef dilation,
                              Tensor& grad_weight);

HICE_API bool conv_bwd_tvm(const Tensor &input, const Tensor &weight, 
                            const Tensor &grad_output,
                            ConstIntArrayRef padding, ConstIntArrayRef stride,
                            ConstIntArrayRef dilation,
                            Tensor& grad_input, Tensor& grad_weight);

HICE_API bool dense_fwd_tvm(const Tensor &input, const Tensor &weight, const Tensor &bias,
                              Tensor &output);
                              
HICE_API bool dense_bwd_tvm(const Tensor &input, const Tensor &weight, const Tensor &grad_output, 
                              Tensor &grad_input, Tensor &grad_weight, Tensor &grad_bias);

HICE_API bool batch_norm_fwd_tvm(const Tensor &input, const Tensor &scale, const Tensor &bias,
                                    const Tensor &running_mean, const Tensor &running_var, 
                                    const Tensor &momentum, const Tensor &eps, 
                                    Tensor &output, Tensor& saved_mean, Tensor& saved_var);

HICE_API bool batch_norm_bwd_tvm(const Tensor &input, const Tensor &scale, const Tensor &saved_mean,
                                    const Tensor &saved_var, const Tensor &eps, 
                                    const Tensor &grad_output,
                                    Tensor &grad_input, Tensor& grad_scale, Tensor& grad_bias);

HICE_API bool relu_fwd_tvm(const Tensor &input, Tensor &output);
                              
// HICE_API bool relu_bwd_tvm(const Tensor &input, Tensor &grad_output, Tensor &grad_input);

} // namespace hice