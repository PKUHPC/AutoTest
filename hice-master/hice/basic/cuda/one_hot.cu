#include "hice/basic/one_hot.h"
#include "hice/device/cuda/context_cuda.h"

namespace hice {

namespace {

__global__ void one_hot_kernel(
    const int64_t outer_size,
    const int64_t axis_size,
    const int64_t inner_size,
    const int64_t* input_data,
    int64_t* output_data) {
  HICE_CUDA_1D_KERNEL_LOOP(index, outer_size * inner_size) {
    int64_t inner_index = index % inner_size;
    int64_t outer_index = index / inner_size;
    int64_t target = input_data[index];
    HICE_CUDA_KERNEL_ASSERT(target >= 0 && target < axis_size);
    int64_t output_index = outer_index * axis_size * inner_size +
                           target * inner_size + inner_index;
    //printf("xx%d\n", output_index);
    output_data[output_index] = 1;
  }
}

void one_hot_impl(const Tensor &input, int64_t num_classes, int64_t axis,
                  Tensor &output) {
  // std::cout << "cuda one hot" << std::endl;
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
  CUDAContext ctx(input.device());
  one_hot_kernel<<<
    cuda_get_1d_num_blocks(outer_size * inner_size),
    cuda_get_1d_num_threads(), 
    0, 
    ctx.stream()>>>(
      outer_size,
      axis_size,
      inner_size,
      input_data,
      output_data);
}

} // anonymous hice

HICE_REGISTER_KERNEL(one_hot_kernel_dispatcher, 
                     &one_hot_impl, 
                     {kCUDA, kDense},
                     {kCUDA, kDense});

} // namespace hice