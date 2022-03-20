#include "hice/core/tensor_impl.h"

#include "hice/core/index_util.h"
#include "hice/core/shape_util.h"

namespace hice {

static void delete_placement_delete_context(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

DataPtr PlacementDeleteContext::make_data_ptr(DataPtr&& data_ptr,
                                              PlacementDtor placement_dtor,
                                              size_t num_items) {
  auto* ptr = data_ptr.get();
  return {ptr,
          new PlacementDeleteContext(std::move(data_ptr), placement_dtor,
                                     num_items),
          &delete_placement_delete_context};
}

TensorImpl::TensorImpl(const TensorOptions& options) {
  shape_.mutable_layout().set_type(options.layout_type());
  storage_ = Storage(0, options.data_type(), options.device(),
                     get_allocator(options.device_type()));
}

TensorImpl::TensorImpl(ConstIntArrayRef dims, const TensorOptions& options) {
  auto min2maj = options.layout().minor_to_major();
  if (dims.size() == min2maj.size()) {
    // Make shape with same layout
    shape_ = ShapeUtil::make_shape_with_layout(dims, min2maj);
  } else {
    // Make shape with default layout. Its type is dense and the minor_to_major is
    // [rank-1, rank-2, rank-3, ..., 0]
    shape_ = ShapeUtil::make_shape(dims);
  }
  // Change the layout type according to the tensor options
  Layout& layout = shape_.mutable_layout();
  layout.set_type(options.layout().type());
  // ShapeUtil::get_num_items return one if dims = {} because the tensor will be
  // a scalar when dims = {} and size_ should be assigned to one.
  size_ = ShapeUtil::get_num_items(shape_);
  offset_ = 0;
  // Right now the storage does not allocate any memory and the memory
  // will be lazily allocated by calling mutable_data() or raw_mutable_data()
  storage_ = Storage(size_, options.data_type(), options.device(),
                     get_allocator(options.device_type()));
}

TensorImpl::TensorImpl(const Shape& shape, Storage storage, int64_t offset)
    : size_(ShapeUtil::get_num_items(shape)),
      offset_(offset),
      shape_(shape),
      storage_(storage){};

void TensorImpl::set_shape(const Shape& new_shape) {
  shape_ = new_shape;
  size_ = ShapeUtil::get_num_items(shape_);
}

int64_t TensorImpl::stride(int64_t dim) const {
  return IndexUtil::get_dim_stride(shape_, dim);
}

std::vector<int64_t> TensorImpl::strides() const {
  return IndexUtil::get_all_strides(shape_);
}

void* TensorImpl::raw_mutable_data(const DataType& dtype) {
  if (data_type() == dtype && storage_initialized()) {
    return static_cast<void*>(static_cast<char*>(storage_.raw_data()) +
                              offset_ * item_size());
  } else {
    HICE_CHECK(has_storage());
    offset_ = 0;
    set_data_type(dtype);
    Allocator* allocator = storage_.allocator();
    if (allocator == nullptr) {
      allocator = hice::get_allocator(device_type());
    }
    // Right now the allocator for CUDA does not support UMA, so it cannot
    // allocate non-trivial data type
    if (data_type().placement_new_fn() && device().type() != DeviceType::CUDA) {
      auto data_ptr = allocator->allocate(size_ * item_size());
      auto dtor = data_type().placement_delete_fn();
      storage_.set_data_ptr(PlacementDeleteContext::make_data_ptr(
          std::move(data_ptr), dtor, size_));
      data_type().placement_new_fn()(storage_.raw_data(), size_);
    } else {
      storage_.set_data_ptr(allocator->allocate(size_ * item_size()));
    }
    storage_.set_size(size_);
    return storage_.raw_data();
  }
}

bool TensorImpl::is_scalar() const {
  return size_ == 1 && ShapeUtil::is_scalar(shape_);
}

bool TensorImpl::is_default_layout() const {
  return LayoutUtil::is_default_layout(shape_);
}

}  // namespace hice
