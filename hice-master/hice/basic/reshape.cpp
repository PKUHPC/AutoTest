#include "hice/basic/reshape.h"
#include "hice/basic/copy.h"
#include "hice/basic/memset.h"
#include "hice/core/shape_util.h"

namespace hice {

Tensor reshape(const Tensor& self, ConstIntArrayRef new_dims) {
  HICE_CHECK(self.is_dense()) << "reshape only supports dense tensor";
  int64_t new_size = hice::size_of_dims(new_dims);
  HICE_CHECK_EQ(self.size(), new_size)
      << "Cannot reshape since the new size and the old size is not equal";
  if (self.is_default_layout()) {
    // Make a new tensor with the default layout by sharing the underlying
    // storage if the layout of the old one is default
    return make_tensor<TensorImpl>(ShapeUtil::make_shape(new_dims),
                                   self.storage(), self.offset());
  } else {
    // Make a new shape with a layout aligned to the old one to make sure not
    // rearrange the underlying storage
    auto new_shape =
        ShapeUtil::align_layouts(self.shape(), ShapeUtil::make_shape(new_dims));
    if (new_shape) {
      return make_tensor<TensorImpl>(new_shape.value(), self.storage(),
                                     self.offset());
    } else {
      // Todo: rearrange the underlying storage by cloning the tensor
      HICE_CHECK(false) << "Cannot set dimensions without rearranging the "
                         "underlying storage";
    }
  }
}

Tensor& reshape_(Tensor& self, ConstIntArrayRef new_dims) {
  HICE_CHECK(self.is_dense()) << "reshape_ only supports dense tensor";
  int64_t new_size = hice::size_of_dims(new_dims);
  HICE_CHECK_EQ(self.size(), new_size)
      << "Cannot reshape since the new size and the old size is not equal";
  if (self.is_default_layout()) {
    // Make a new tensor with the default layout by sharing the underlying
    // storage if the layout of the old one is default
    self.mutable_impl().set_shape(ShapeUtil::make_shape(new_dims));
    return self;
  } else {
    // Make a new shape with a layout aligned to the old one to make sure not
    // rearrange the underlying storage
    auto new_shape =
        ShapeUtil::align_layouts(self.shape(), ShapeUtil::make_shape(new_dims));
    if (new_shape) {
      self.mutable_impl().set_shape(new_shape.value());
      return self;
    } else {
      // Todo: rearrange the underlying storage by cloning the tensor
      HICE_CHECK(false) << "Cannot set dimensions without rearranging the "
                         "underlying storage";
    }
  }
}

Tensor expand_dims(const Tensor& self, int64_t axis) {
  HICE_CHECK(self.is_dense()) << "expand_dims only supports dense tensor";
  int64_t true_axis = hice::get_true_axis(axis, self.ndim() + 1);
  Shape shape = self.shape();
  // Step 1: Insert the 1-size dimension at the position indexed by axis
  auto& dims = shape.mutable_dimensions();
  dims.insert(dims.begin() + true_axis, 1);
  auto& min2maj = shape.mutable_layout().mutable_minor_to_major();
  // Step 2: Adjust the corresponding minor_to_major. We first record the
  // original position for later insertion in the min2maj and then increment the
  // logical dims by one which are greater and equal than the true axis. When
  // the true_axis is after the last original dim, we cannot find the true axis
  // in the min2maj, so we locate the position of (true_axis - 1)
  int64_t pos = 0; // Initialized to 0 for the scalar case
  int64_t val = (true_axis == self.ndim()) ? true_axis - 1 : true_axis;
  // Don't change the order of the following two loop
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] == val) pos = i;
  }
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] >= true_axis) min2maj[i] += 1;
  }
  // The true axis should be inserted before the the value (true_axis - 1) in
  // the min2maj if it is after the last original dim according to our defaut
  // minor-to-major order
  if (true_axis ==  self.ndim()) pos -= 1;
#if 0
  std::cout << "true axis: " << true_axis << " pos: " << pos << std::endl;
  for (int i : min2maj) std::cout << i << ", ";
  std::cout << std::endl;
#endif
  // Insert the true axis in the new adjusted min2maj at the corret position
  min2maj.insert(min2maj.begin() + pos + 1, true_axis);
  return make_tensor<TensorImpl>(shape, self.storage(), self.offset());
}

Tensor& expand_dims_(Tensor& self, int64_t axis) {
  HICE_CHECK(self.is_dense()) << "expand_dims_ only supports dense tensor";
  int64_t true_axis = hice::get_true_axis(axis, self.ndim() + 1);
  int64_t old_ndim = self.ndim();
  Shape& shape = self.mutable_impl().mutable_shape();
  // Step 1: Insert the 1-size dimension at the position indexed by axis
  auto& dims = shape.mutable_dimensions();
  dims.insert(dims.begin() + true_axis, 1);
  auto& min2maj = shape.mutable_layout().mutable_minor_to_major();
  // Step 2: Adjust the corresponding minor_to_major. We first record the
  // original position for later insertion in the min2maj and then increment the
  // logical dims by one which are greater and equal than the true axis. When
  // the true_axis is after the last original dim, we cannot find the true axis
  // in the min2maj, so we locate the position of (true_axis - 1)
  int64_t pos = 0; // Initialized to 0 for the scalar case
  int64_t val = (true_axis == old_ndim) ? true_axis - 1 : true_axis;
  // Don't change the order of the following two loop
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] == val) pos = i;
  }
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] >= true_axis) min2maj[i] += 1;
  }
  // The true axis should be inserted before the the value (true_axis - 1) in
  // the min2maj if it is after the last original dim according to our defaut
  // minor-to-major order
  if (true_axis == old_ndim) pos -= 1;
  // Insert the true axis in the new adjusted min2maj at the corret position
  min2maj.insert(min2maj.begin() + pos + 1, true_axis);
  return self;
}

Tensor squeeze(const Tensor& self, int64_t axis) {
  HICE_CHECK(self.is_dense()) << "squeeze only supports dense tensor";
  int64_t true_axis = hice::get_true_axis(axis, self.ndim());
  if (self.ndim() == 0 || self.dim(true_axis) != 1) return self;
  Shape shape = self.shape();
  // Step 1: Insert the 1-size dimension at the position indexed by axis
  auto& dims = shape.mutable_dimensions();
  dims.erase(dims.begin() + true_axis);
  auto& min2maj = shape.mutable_layout().mutable_minor_to_major();
  // Step 2: Adjust the corresponding minor_to_major. We first record the
  // original position for later deletion in the min2maj and then decrement the
  // logical dims by one which are greater and equal than the true axis.
  int64_t pos = 0; // Initialized to 0 for the scalar case
  // Don't change the order of the following two loop
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] == true_axis) pos = i;
  }
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] > true_axis) min2maj[i] -= 1;
  }
  // Delete the true axis in the new adjusted min2maj at the corret position
  min2maj.erase(min2maj.begin() + pos);
  return make_tensor<TensorImpl>(shape, self.storage(), self.offset());
}

Tensor& squeeze_(Tensor& self, int64_t axis) {
  HICE_CHECK(self.is_dense()) << "squeeze_ only supports dense tensor";
  int64_t true_axis = hice::get_true_axis(axis, self.ndim());
  if (self.ndim() == 0 || self.dim(true_axis) != 1) return self;
  Shape& shape = self.mutable_impl().mutable_shape();
  // Step 1: Insert the 1-size dimension at the position indexed by axis
  auto& dims = shape.mutable_dimensions();
  dims.erase(dims.begin() + true_axis);
  auto& min2maj = shape.mutable_layout().mutable_minor_to_major();
  // Step 2: Adjust the corresponding minor_to_major. We first record the
  // original position for later deletion in the min2maj and then decrement the
  // logical dims by one which are greater and equal than the true axis.
  int64_t pos = 0; // Initialized to 0 for the scalar case
  // Don't change the order of the following two loop
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] == true_axis) pos = i;
  }
  for (int64_t i = 0; i < min2maj.size(); ++i) {
    if (min2maj[i] > true_axis) min2maj[i] -= 1;
  }
  // Delete the true axis in the new adjusted min2maj at the corret position
  min2maj.erase(min2maj.begin() + pos);
  return self;
}

Tensor contiguous(const Tensor& self) {
  HICE_CHECK(self.is_dense()) << "contiguous only supports dense tensor";
  if (self.is_default_layout()) {
    return self;
  }
  Tensor tensor_new(self.dims(), device(self.device()).dtype(self.data_type()));
  copy(self, tensor_new);
  return tensor_new;
}

Tensor& contiguous_(Tensor& self) {
  Tensor self_contgs = contiguous(self);
  self.mutable_impl().set_shape(self_contgs.shape());
  self.mutable_impl().set_storage(self_contgs.storage());
  return self;
}

} // namespace hice