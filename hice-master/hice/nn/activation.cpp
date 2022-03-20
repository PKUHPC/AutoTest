#include "hice/nn/activation.h"
#include "hice/core/scalar_ops.h"

namespace hice {

// Forward
HICE_DEFINE_DISPATCHER(abs_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(relu_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(sigmoid_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(sqrt_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(square_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(tanh_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(elu_fwd_dispatcher);

Tensor abs_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  abs_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor relu_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  relu_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor sigmoid_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  sigmoid_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor sqrt_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  sqrt_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor square_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  square_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor tanh_fwd(const Tensor& input) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  tanh_fwd_dispatcher(input, output, /* resizable = */true);
  return output;
}

Tensor elu_fwd(const Tensor& input, Scalar alpha) {
  Tensor output(device(input.device()).dtype(input.data_type()));
  Tensor alpha_t = scalar_to_tensor(alpha, 
                                    input.scalar_type(),
                                    input.device_type());
  elu_fwd_dispatcher(input, alpha_t, output, /* resizable = */true);
  return output;
}

// Inplace
Tensor& abs_fwd(const Tensor& input, Tensor& output) {
  abs_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& relu_fwd(const Tensor& input, Tensor& output) {
  // std::cout<<"hice relu fwd"<<std::endl;
  relu_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& sigmoid_fwd(const Tensor& input, Tensor& output) {
  sigmoid_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& sqrt_fwd(const Tensor& input, Tensor& output) {
  sqrt_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& square_fwd(const Tensor& input, Tensor& output) {
  square_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& tanh_fwd(const Tensor& input, Tensor& output) {
  tanh_fwd_dispatcher(input, output, /* resizable = */false);
  return output;
}

Tensor& elu_fwd(const Tensor& input, Scalar alpha, Tensor& output) {
  Tensor alpha_t = scalar_to_tensor(alpha, 
                                    input.scalar_type(),
                                    input.device_type());
  elu_fwd_dispatcher(input, alpha_t, output, /* resizable = */false);
  return output;
}


// Backward
HICE_DEFINE_DISPATCHER(abs_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(relu_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(sigmoid_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(sqrt_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(square_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(tanh_bwd_dispatcher);
HICE_DEFINE_DISPATCHER(elu_bwd_dispatcher);
// HICE_DEFINE_DISPATCHER(elu_bwd_dispatcher);

Tensor abs_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  abs_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor relu_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  relu_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor sigmoid_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  sigmoid_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor sqrt_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  sqrt_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor square_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  square_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor tanh_bwd(const Tensor& input, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  tanh_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

Tensor elu_bwd(const Tensor& input, Scalar alpha, const Tensor& grad_output) {
  Tensor grad_input(device(input.device()).dtype(input.data_type()));
  Tensor alpha_t = scalar_to_tensor(alpha, 
                                    input.scalar_type(),
                                    input.device_type());
  elu_bwd_dispatcher(input, alpha_t, grad_output, grad_input, /* resizable = */true);
  return grad_input;
}

// Inplace
Tensor& abs_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  abs_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& relu_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  // std::cout<<"hice relu bwd"<<std::endl;
  relu_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& sigmoid_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  sigmoid_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& sqrt_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  sqrt_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& square_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  square_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& tanh_bwd(const Tensor& input, const Tensor& grad_output, Tensor& grad_input) {
  tanh_bwd_dispatcher(input, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

Tensor& elu_bwd(const Tensor& input, Scalar alpha, const Tensor& grad_output,
                Tensor& grad_input) {
  Tensor alpha_t = scalar_to_tensor(alpha, 
                                    input.scalar_type(),
                                    input.device_type());
  elu_bwd_dispatcher(input, alpha_t, grad_output, grad_input, /* resizable = */false);
  return grad_input;
}

} // namespace hice
