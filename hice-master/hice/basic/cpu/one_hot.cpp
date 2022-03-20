#include "hice/basic/one_hot.h"

namespace hice {

namespace {

void one_hot_impl(const Tensor &input, int64_t num_classes, int64_t axis,
                  Tensor &output) {
  //std::cout << "cpu one hot" << std::endl;
  int64_t true_axis = output.get_true_axis(axis);
  int64_t outer_size = output.size_to_dim(true_axis);
  int64_t axis_size = output.dim(true_axis);
  int64_t inner_size = output.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(axis_size, num_classes);
  HICE_CHECK_EQ(input.scalar_type(), kInt64);
  HICE_CHECK_EQ(output.scalar_type(), kInt64);
  const int64_t *input_data = input.data<int64_t>();
  int64_t *output_data = output.mutable_data<int64_t>();
  output.fill(0); // Todo:: Use memset to improve the performance
  //memset(output_data, 0, output.size() * item_size());
  for (int64_t index = 0; index < outer_size * inner_size; ++index) {
    HICE_CHECK_GE(input_data[index], 0);
    HICE_CHECK_LT(input_data[index], axis_size);
    int64_t inner_index = index % inner_size;
    int64_t outer_index = index / inner_size;
    int64_t output_index = outer_index * axis_size * inner_size +
                           input_data[index] * inner_size + inner_index;
    output_data[output_index] = 1;
  }
}

} // anonymous namespace

HICE_REGISTER_KERNEL(one_hot_kernel_dispatcher, 
                     &one_hot_impl, 
                     {kCPU, kDense},
                     {kCPU, kDense});

} // namespace hice