#include "hice/basic/split.h"
#include "hice/basic/factories.h"
#include "hice/basic/reshape.h"
#include "hice/basic/slice.h"
#include "hice/core/layout_util.h"
#include "hice/core/shape_util.h"

namespace hice {

std::vector<Tensor> split_with_sizes(const Tensor& self, int64_t axis,
                                     ConstIntArrayRef sizes) {
  HICE_CHECK(self.is_dense()) << "split only supports dense tensor";
  HICE_CHECK(LayoutUtil::is_default_layout(self.shape()));
  HICE_CHECK_EQ(self.offset(), 0);
  HICE_CHECK_GT(sizes.size(), 1);
  int64_t true_axis = self.get_true_axis(axis);
  int64_t num_tensors = sizes.size();
  int64_t total_num = 0;
  for (int i = 0; i < num_tensors; ++i) {
    total_num += sizes[i];
  }
  HICE_CHECK_EQ(total_num, self.dim(true_axis));
  std::vector<Tensor> tensor_list;
  tensor_list.reserve(num_tensors);
  // call slice
  int64_t start = 0;
  int64_t end = 0;
  for (int64_t i = 0; i < num_tensors; ++i) {
    end += sizes[i];
    // std::cout<<"start="<<start<<", end="<<end<<std::endl;
    Tensor result = hice::slice(self, axis, start, end);
    tensor_list.push_back(result);
    start = end;
  }
  return tensor_list;
}

std::vector<Tensor> split(const Tensor& self, int64_t axis,
                          int64_t num_tensors) {
  HICE_CHECK_GT(num_tensors, 1);
  HICE_CHECK_EQ(self.dim(axis) % num_tensors, 0);
  std::vector<int64_t> sizes(num_tensors, self.dim(axis) / num_tensors);
  return split_with_sizes(self, axis, sizes);
}

}  // namespace hice
