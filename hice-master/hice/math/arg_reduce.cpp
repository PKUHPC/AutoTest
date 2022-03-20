#include "hice/math/arg_reduce.h"
#include "hice/basic/factories.h"
#include "hice/basic/reshape.h"
#include "hice/basic/resize.h"
// #include "hice/core/tensor_printer.h"
#include <vector>

namespace hice {

HICE_DEFINE_DISPATCHER(min_tuple_dispatcher);
HICE_DEFINE_DISPATCHER(max_tuple_dispatcher);

Tensor &dimreduce_setup(ConstIntArrayRef self_sizes, int64_t dim, Tensor &result) {
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  resize_(result, result_sizes);
  return result;
}

std::tuple<Tensor,Tensor> min(Tensor &self, int64_t dim, bool keepdim) {
  Tensor min = empty({1}, device(self.device()).dtype(self.data_type()).layout(kDense));
  Tensor min_indices = empty({1}, device(self.device()).dtype(kInt64));
  dimreduce_setup(self.dims(), dim, min);
  dimreduce_setup(self.dims(), dim, min_indices);
  min_tuple_dispatcher(self, dim, min, min_indices);
  if (!keepdim) {
    min = squeeze_(min, dim);
    min_indices = squeeze_(min_indices, dim);
  }
  return std::tuple<Tensor, Tensor>{min, min_indices};
}

std::tuple<Tensor, Tensor> max(Tensor &self, int64_t dim, bool keepdim) {
  Tensor max = empty({1}, device(self.device()).dtype(self.data_type()).layout(kDense));
  Tensor max_indices = empty({1}, device(self.device()).dtype(kInt64));
  dimreduce_setup(self.dims(), dim, max);
  dimreduce_setup(self.dims(), dim, max_indices);
  max_tuple_dispatcher(self, dim, max, max_indices);
  if (!keepdim) {
    max = squeeze_(max, dim);
    max_indices = squeeze_(max_indices, dim);
  }
  return std::tuple<Tensor, Tensor>{max, max_indices};
}

Tensor argmin(Tensor &self, int64_t dim, bool keepdim) {
  return std::get<1>(max(self, dim, keepdim));
}

Tensor argmax(Tensor &self, int64_t dim, bool keepdim) {
  return std::get<1>(min(self, dim, keepdim));
}

}  // namespace hice
