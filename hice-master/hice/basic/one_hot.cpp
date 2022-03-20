#include "hice/basic/one_hot.h"

namespace hice {

HICE_DEFINE_DISPATCHER(one_hot_kernel_dispatcher);

Tensor one_hot(const Tensor &input, int64_t num_classes, int64_t axis) {
  std::vector<int64_t> output_dims(input.dims().begin(), input.dims().end());
  int64_t true_axis = hice::get_true_axis(axis, input.ndim() + 1);
  output_dims.insert(output_dims.begin() + true_axis, num_classes);
  Tensor output(output_dims, device(input.device())
                                 .dtype(input.data_type())
                                 .layout(input.layout_type()));
  one_hot_kernel_dispatcher(input, num_classes, axis, output);
  return output;
}

} // namespace hice