#pragma once 

#include <tuple>

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"

namespace hice {

const static int HICE_BATCHNORM_PER_ACTIVATION = 1;
const static int HICE_BATCHNORM_SPATIAL = 2;

// Dispatcher
using batch_norm_fwd_kernel_fn_type = void (*)(
	  Tensor & input, Tensor & output,
    Tensor & bn_scale, Tensor & bn_bias,
    Tensor & running_mean, Tensor & running_var,
    bool train, int mode, double epsilon, double expo_factor,
    Tensor & saved_mean, Tensor & saved_inv_var);
HICE_DECLARE_DISPATCHER(batch_norm_fwd_dispatcher, batch_norm_fwd_kernel_fn_type);

using batch_norm_bwd_kernel_fn_type = void (*)(
	Tensor & input, Tensor & output_grad,
    Tensor & bn_scale, Tensor & bn_bias,
    Tensor & saved_mean, Tensor & saved_inv_var,
    int mode, double epsilon, 
    Tensor & bn_scale_grad, Tensor & bn_bias_grad,
    Tensor & input_grad);
HICE_DECLARE_DISPATCHER(batch_norm_bwd_dispatcher, batch_norm_bwd_kernel_fn_type);

// Operators
HICE_API std::tuple<Tensor, Tensor, Tensor> batch_norm_fwd(
    Tensor &input, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
    Tensor &running_var, bool training, uint32_t mode,
    double exponential_average_factor, double epsilon);
HICE_API std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_fwd(
    Tensor &input, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
    Tensor &running_var, bool training, uint32_t mode,
    double exponential_average_factor, double epsilon, Tensor& output, Tensor& saved_mean, Tensor& saved_inv_var);


HICE_API std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd(
    Tensor &input, Tensor &output_grad, Tensor &bn_scale, Tensor &bn_bias,
    Tensor &saved_mean, Tensor &saved_inv_var, uint32_t mode, double epsilon);
HICE_API std::tuple<Tensor&,Tensor&,Tensor&> batch_norm_bwd(
  Tensor &input, Tensor &output_grad,
  Tensor & bn_scale, Tensor & bn_bias,
  Tensor & saved_mean, Tensor & saved_inv_var,
  uint32_t mode, double epsilon, Tensor& input_grad, Tensor& bn_scale_grad, Tensor& bn_bias_grad);

} // namesapce hice
