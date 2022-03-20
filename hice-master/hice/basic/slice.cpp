#include "hice/basic/slice.h"
#include "hice/basic/factories.h"
#include "hice/core/shape_util.h"

namespace hice {

Tensor slice(const Tensor& self, int64_t axis, int64_t start, int64_t end) {
  HICE_CHECK(self.is_dense()) << "slice only supports dense tensor";
  auto dims = self.dims();
  HICE_CHECK_GE(dims.size(), 0);
  int64_t true_axis = self.get_true_axis(axis);
  if (start < 0) {
    start += dims[true_axis];
  }
  if (end < 0) {
    end += dims[true_axis];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= dims[true_axis]) {
    start = dims[true_axis];
  }
  if (end < start) {
    end = start;
  } else if (end >= dims[true_axis]) {
    end = dims[true_axis];
  }
  std::vector<int64_t> dims_result(self.dims().begin(), self.dims().end());
  dims_result[true_axis] = end - start;
  if (true_axis == 0) {
    // share storage
    int64_t offset = self.stride(0) * start;
    Shape shape_result = ShapeUtil::make_shape(dims_result);
    // std::cout<<"offset="<<offset<<", start="<<start<<", end="<<end<<std::endl;
    return make_tensor<TensorImpl>(shape_result, self.storage(), offset);
  } else {
    // copy data
    Tensor result(dims_result, self.options());
    int64_t outer_size = self.size_to_dim(true_axis);
    int64_t inner_size = self.size_from_dim(true_axis + 1);
    size_t item_size = self.data_type().size();
    int64_t copy_stride_self_bytes = inner_size * dims[true_axis] * item_size;
    int64_t copy_stride_result_bytes = inner_size * dims_result[true_axis] * item_size;
    int64_t offset_self_bytes = item_size * start * inner_size;
    int64_t offset_result_bytes = 0;
    int64_t copy_size_bytes = copy_stride_result_bytes;
    // std::cout<<"item_size="<<item_size<<std::endl;
    // std::cout<<"outer_size="<<outer_size<<std::endl;
    // std::cout<<"inner_size="<<inner_size<<std::endl;
    for (int64_t i = 0; i < outer_size; ++i) {
      // std::cout<<"outer_i="<<i<<", offset_self_bytes="<<offset_self_bytes <<", offset_result_bytes="<<offset_result_bytes<<std::endl;
      hice::copy_bytes(copy_size_bytes, (char *)self.raw_data() + offset_self_bytes,
                       self.device(), (char *)result.raw_mutable_data() + offset_result_bytes,
                       result.device());
      offset_self_bytes += copy_stride_self_bytes;
      offset_result_bytes += copy_stride_result_bytes;
    }
    return result;
  }
  // return Tensor();
}

}  // namespace hice
