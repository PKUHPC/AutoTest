#include "hice/basic/transpose.h"
#include "hice/core/shape_util.h"

namespace hice {

Tensor transpose(const Tensor& tensor, ConstIntArrayRef perm, bool conjugate) {
  Tensor tensor_new = make_tensor<TensorImpl>(tensor.shape(), tensor.storage(),
                                     tensor.offset());
  return transpose_(tensor_new, perm, conjugate);
}

Tensor& transpose_(Tensor& tensor, ConstIntArrayRef perm, bool conjugate) {
  HICE_CHECK(tensor.is_dense()) << "transpose only supports dense tensor";
  HICE_CHECK_EQ(conjugate, false);  // conjugate is not support for now.
  auto ndim = tensor.ndim();
  auto& shape = tensor.mutable_impl().mutable_shape();
  auto dims_old = shape.dimensions();
  auto layout_old = shape.layout().minor_to_major();
  std::vector<int64_t> dims_new(ndim);
  std::vector<int64_t> layout_new(ndim);
  // make shape from perm
  if (perm.size() == 0) {
    for (int i = 0; i < ndim; ++i) {
      dims_new[i] = dims_old[ndim - 1 - i];
      layout_new[i] = layout_old[ndim - 1 - i];
    }
  } else {
    HICE_CHECK_EQ(perm.size(), ndim);
    std::vector<bool> mark(ndim, false);
    for (int i = 0; i < ndim; ++i) {
      dims_new[i] = dims_old[perm[i]];
      layout_new[i] = layout_old[perm[i]];
      mark[perm[i]] = true;
    }
    for (int i = 0; i < ndim; ++i) {
      HICE_CHECK_EQ(mark[i], true);
    }
  }
  auto shape_new = ShapeUtil::make_shape_with_layout(dims_new, layout_new);
  shape.swap(shape_new);
  return tensor;
}

Tensor transpose_matrix(const Tensor& tensor, bool conjugate) {
  Tensor tensor_new = make_tensor<TensorImpl>(tensor.shape(), tensor.storage(),
                                     tensor.offset());
  return transpose_matrix_(tensor_new, conjugate);
}

Tensor& transpose_matrix_(Tensor& tensor, bool conjugate) {
  int64_t ndim = tensor.ndim();
  std::vector<int64_t> perm(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    perm[i] = i;
  }
  if (ndim >= 2) {
    std::swap(perm[ndim - 1], perm[ndim - 2]);
  }
  return transpose_(tensor, perm, conjugate);
}

} // namespce hice



// The following version of transpose is implemented by Ye Zilingfeng.
#if 0
#include "hice/basic/memset.h"

namespace hice {

HICE_DEFINE_DISPATCHER(transpose_dispatcher);

/* -- Out-of-place --*/
HICE_API Tensor transpose(const Tensor& input, 
                          ConstIntArrayRef perm_dims){
  Tensor output(input.dims(), device(input.device())
                                .dtype(input.data_type())
                                .layout(input.layout_type()));
  transpose_dispatcher(input, perm_dims, output);
  return output;
}

/* -- In-place --*/
HICE_API Tensor& transpose_(const Tensor& input, 
                            ConstIntArrayRef perm_dims, 
                            Tensor & output){
  transpose_dispatcher(input, perm_dims, output);
  return output;
}

} // namespace hice

#endif