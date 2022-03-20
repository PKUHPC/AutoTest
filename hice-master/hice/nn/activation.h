#pragma once

// NOTE: the precision is not very high for now.

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"

namespace hice {

// Forward Dispatcher
using activation_fwd_kernel_fn_type = void (*)(const Tensor& input, 
                                              Tensor& output,
                                              bool output_resizable);
using elu_fwd_kernel_fn_type = void (*)(const Tensor& input, 
                                        const Tensor& alpha,
                                        Tensor& output,
                                        bool output_resizable);

HICE_DECLARE_DISPATCHER(abs_fwd_dispatcher    , activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(relu_fwd_dispatcher   , activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(sigmoid_fwd_dispatcher, activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(sqrt_fwd_dispatcher   , activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(square_fwd_dispatcher , activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(tanh_fwd_dispatcher   , activation_fwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(elu_fwd_dispatcher    , elu_fwd_kernel_fn_type);

// Operators
HICE_API Tensor abs_fwd(const Tensor& input);
HICE_API Tensor relu_fwd(const Tensor& input);
HICE_API Tensor sigmoid_fwd(const Tensor& input);
HICE_API Tensor sqrt_fwd(const Tensor& input);
HICE_API Tensor square_fwd(const Tensor& input);
HICE_API Tensor tanh_fwd(const Tensor& input);
HICE_API Tensor elu_fwd(const Tensor& input, Scalar alpha);

// Inplace
HICE_API Tensor& abs_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& relu_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& sigmoid_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& sqrt_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& square_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& tanh_fwd(const Tensor& input, Tensor& output);
HICE_API Tensor& elu_fwd(const Tensor& input, Scalar alpha, Tensor& output);


// Backward Dispatcher
using activation_bwd_kernel_fn_type = void (*)(const Tensor& input,
                                              const Tensor& grad_output,
                                              Tensor& grad_input,
                                              bool output_resizable);
using elu_bwd_kernel_fn_type = void (*)(const Tensor& input,
                                        const Tensor& alpha,
                                        const Tensor& grad_output,
                                        Tensor& grad_input,
                                        bool output_resizable);
                                           
HICE_DECLARE_DISPATCHER(abs_bwd_dispatcher    , activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(relu_bwd_dispatcher   , activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(sigmoid_bwd_dispatcher, activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(sqrt_bwd_dispatcher   , activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(square_bwd_dispatcher , activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(tanh_bwd_dispatcher   , activation_bwd_kernel_fn_type);
HICE_DECLARE_DISPATCHER(elu_bwd_dispatcher   , elu_bwd_kernel_fn_type);

HICE_API Tensor abs_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor relu_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor sigmoid_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor sqrt_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor square_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor tanh_bwd(const Tensor& input, const Tensor& grad_output);
HICE_API Tensor elu_bwd(const Tensor& input, Scalar alpha, const Tensor& grad_output);

// Inplace
HICE_API Tensor& abs_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& relu_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& sigmoid_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& sqrt_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& square_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& tanh_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input);
HICE_API Tensor& elu_bwd(const Tensor& input, Scalar alpha, 
                         const Tensor& grad_output,
                         Tensor& grad_input);





} // namespace hice
