#include "hice/math/unary_expr.h"

namespace hice {

HICE_DEFINE_DISPATCHER(exp_dispatcher);
HICE_DEFINE_DISPATCHER(log_dispatcher);
HICE_DEFINE_DISPATCHER(neg_dispatcher);

// outplace
Tensor exp(const Tensor& tensor) {
  Tensor result(device(tensor.device()).dtype(tensor.data_type()));
  exp_dispatcher(tensor, result, /* resizable = */true);
  return result;
}

Tensor log(const Tensor& tensor) {
  Tensor result(device(tensor.device()).dtype(tensor.data_type()));
  log_dispatcher(tensor, result, /* resizable = */true);
  return result;
}

Tensor neg(const Tensor& tensor) {
  Tensor result(device(tensor.device()).dtype(tensor.data_type()));
  neg_dispatcher(tensor, result, /* resizable = */true);
  return result;
}

// inplace
Tensor& exp(const Tensor& tensor, Tensor& result) {
  exp_dispatcher(tensor, result, /* resizable = */false);
  return result;
}

Tensor& log(const Tensor& tensor, Tensor& result) {
  log_dispatcher(tensor, result, /* resizable = */false);
  return result;
}

Tensor& neg(const Tensor& tensor, Tensor& result) {
  neg_dispatcher(tensor, result, /* resizable = */false);
  return result;
}

} // namespce hice
