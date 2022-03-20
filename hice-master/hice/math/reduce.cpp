#include "hice/math/reduce.h"

namespace hice {

std::vector<int64_t> get_all_dims_range(ConstIntArrayRef dims) {
  std::vector<int64_t> range;
  for (int64_t i = 0; i < dims.size(); ++i) {
    range.push_back(i);
  }
  return range;
}

HICE_DEFINE_DISPATCHER(reduce_sum_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_prod_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_mean_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_and_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_or_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_min_dispatcher);
HICE_DEFINE_DISPATCHER(reduce_max_dispatcher);

Tensor reduce_sum(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_sum_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_prod(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_prod_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_mean(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_mean_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_and(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_and_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_or(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_or_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_min(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_min_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor reduce_max(const Tensor &self, ConstIntArrayRef dim, bool keepdim) {
  Tensor result(device(self.device()).dtype(self.data_type()).layout(kDense));
  reduce_max_dispatcher(self, dim, keepdim, result, /* resizable = */true);
  return result;
}

Tensor& reduce_sum(const Tensor &self, ConstIntArrayRef dim, bool keepdim, Tensor &output) {
  reduce_sum_dispatcher(self, dim, keepdim, output, /* resizable = */false);
  return output;
}

Tensor& reduce_mean(const Tensor &self, ConstIntArrayRef dim, bool keepdim, Tensor &output) {
  reduce_mean_dispatcher(self, dim, keepdim, output, /* resizable = */false);
  return output;
}

}  // namespace hice
