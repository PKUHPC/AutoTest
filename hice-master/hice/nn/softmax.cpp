#include "hice/nn/softmax.h"

namespace hice {

HICE_DEFINE_DISPATCHER(softmax_fwd_dispatcher);

Tensor softmax_fwd(const Tensor &input, int64_t axis) {
  Tensor output(
      input.dims(),
      device(input.device()).dtype(input.data_type()).layout(input.layout()));
  softmax_fwd_dispatcher(input, axis, output);
  return output;
}

Tensor& softmax_fwd(const Tensor &input, int64_t axis, Tensor &output) {
  softmax_fwd_dispatcher(input, axis, output);
  return output;
}

HICE_DEFINE_DISPATCHER(softmax_bwd_dispatcher);

Tensor softmax_bwd(const Tensor &output, const Tensor &grad_output,
                   int64_t axis) {
  Tensor grad_input(output.dims(), device(output.device())
                                       .dtype(output.data_type())
                                       .layout(output.layout()));
  softmax_bwd_dispatcher(output, grad_output, axis, grad_input);
  return grad_input;
}

Tensor &softmax_bwd(const Tensor &output, const Tensor &grad_output,
                    int64_t axis, Tensor &grad_input) {
  softmax_bwd_dispatcher(output, grad_output, axis, grad_input);
  return grad_input;
}

} // namespace hice
