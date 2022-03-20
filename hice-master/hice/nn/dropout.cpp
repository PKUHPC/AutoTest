#include "dropout.h"

namespace hice {

HICE_DEFINE_DISPATCHER(dropout_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(dropout_bwd_dispatcher);

Tensor dropout_fwd(Tensor &input, double rate, Tensor &mask) {
  Tensor result(input.dims(),
                device(input.device()).dtype(input.data_type()).layout(kDense));
  dropout_fwd_dispatcher(input, rate, mask, result);
  return result;
}

Tensor& dropout_fwd(Tensor &input, double rate, Tensor &mask, Tensor& output) {
  dropout_fwd_dispatcher(input, rate, mask, output);
  return output;
}

Tensor dropout_bwd(Tensor &input, double rate, Tensor &mask) {
  Tensor result(input.dims(),
                device(input.device()).dtype(input.data_type()).layout(kDense));
  dropout_bwd_dispatcher(input, rate, mask, result);
  return result;
}

} // namespace hice
