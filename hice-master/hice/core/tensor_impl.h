#pragma once

#include <iostream>

#include "hice/util/type_id.h"
#include "hice/util/intrusive_ptr.h"
#include "hice/util/copy_bytes.h"
#include "hice/util/loguru.h"
#include "hice/core/tensor_options.h"
#include "hice/core/scalar_type.h"
#include "hice/core/scalar.h"
#include "hice/core/device.h"
#include "hice/core/dimension.h"
#include "hice/core/allocator.h"
#include "hice/core/shape.h"
#include "hice/core/storage.h"

namespace hice {

using PlacementDtor = void (*)(void*, size_t);
struct PlacementDeleteContext {
  DataPtr data_ptr_;
  PlacementDtor placement_dtor_;
  size_t size_;

  PlacementDeleteContext(DataPtr&& data_ptr, PlacementDtor placement_dtor,
                         size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}

  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // Original memory will be freed when data_ptr_ member is destructed
  }

  static DataPtr make_data_ptr(DataPtr&& data_ptr, PlacementDtor placement_dtor,
                               size_t size);
};

/// The low-level implementation of a tensor. It consists of a metadata shape (
/// which constains the dimensions and layout) and a storage (which contains the
/// actual data). The shape is used to interprate the underlying storage.
/// Multiple TensorImpl can share the same storage with different metadata. 
class HICE_API TensorImpl: public intrusive_ptr_target {
 public:
  /// Construct an empty TensorImpl(dims=[], size_=0) by specifying the meta data.
  TensorImpl(const TensorOptions& options);

  /// Construct TensorImpl by specifying the meta data. This method does not
  /// allocate the memory. To trigger the allocation, users should call
  /// mutable_data() or raw_mutable_data(). By using the lazy evaluation, we can
  /// adjust the metadata with a low cost before using it. 
  /// Note that when dims = {}, the tensor should be a scalar whose size_ == 1 and
  /// shape.rank() == 0; when dims = {0}, the tensor is a vector with zero
  /// item whose size_ == 0 but shape_.rank() == 1; when dims = {1}, the tensor is
  /// a vector with one item whose size_ == 1 and shape_.rank() == 1.
  TensorImpl(ConstIntArrayRef dims, const TensorOptions& options);

  /// Construct TensorImpl by using the shape and sharing the storage. This
  /// method should be only used by developers, who can make sure the shape and
  /// the storage are compatible.
  TensorImpl(const Shape &shape, Storage storage, int64_t offset = 0);

  TensorImpl(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl& operator=(TensorImpl&&) = default;

  int64_t size() const { return size_; }
  int64_t rank() const { return shape_.rank(); }
  int64_t ndim() const { return shape_.dimensions_size(); }
  int64_t dim(int64_t d) const { return shape_.dimensions()[d]; }
  ConstIntArrayRef dims() const { return shape_.dimensions(); }

  // The stride information will be computed again right now based on the
  // current shape_ since the shape_ may be change by users
  virtual int64_t stride(int64_t d) const;
  virtual std::vector<int64_t> strides() const; 

  virtual int64_t offset() const { return offset_; }
  virtual void set_offset(int64_t offset) { offset_ = offset; }

  DeviceType device_type() const { return device().type(); }
  Device device() const { return storage_.device(); }

  ScalarType scalar_type() const { return DataTypeToScalarType(data_type()); }
  const DataType& data_type() const { return storage_.data_type(); }
  virtual void set_data_type(const DataType &dtype) { storage_.set_data_type(dtype); }
  size_t item_size() const { return data_type().size(); }

  const Shape& shape() const { return shape_; }
  Shape& mutable_shape() { return shape_; }
  // Set a new shape and also update the size_ according to the new shape.
  // Callers should be responsible for making sure the new shape is compatible
  // to the underlying storage.
  void set_shape(const Shape& shape); 
  LayoutType layout_type() const { return layout().type(); }
  const Layout& layout() const { return shape_.layout(); }

  virtual const Storage& storage() const { return storage_; }
  virtual Storage& mutable_storage() { return storage_; }
  virtual void set_storage(Storage storage) { storage_ = std::move(storage); }
  virtual bool has_storage() const { return storage_; }
  bool storage_initialized() const {
    HICE_CHECK(has_storage()) << "Cannot call storage_initialized on tensor "
                                 "that does not have storage";
    return (storage_.raw_data()|| size_ == 0);
  }

  template <typename T>
  T* data() const {
    HICE_CHECK(storage_initialized());
    HICE_CHECK(storage_.is_type<T>())
        << "Tensor type mismatch, caller expects elements to be "
        << DataType::name<T>() << ", while tensor contains "
        << data_type().name() << ". ";
    return storage_.data<T>() + offset_;
  }
  void* raw_data() const {
    HICE_CHECK(storage_initialized());
    return static_cast<void*>(static_cast<char*>(storage_.raw_data()) +
                              item_size() * offset_);
  }
  template <typename T>
  T* mutable_data() {
    if (storage_initialized() && storage_.is_type<T>()) {
      return static_cast<T*>(storage_.raw_data()) + offset_;
    }
    // Check it here statically - otherwise TypeMeta would throw the runtime
    // error in attempt to invoke TypeMeta::ctor()
    static_assert(std::is_default_constructible<T>::value,
                  "Tensor can't hold non-default-constructible types");
    return static_cast<T*>(raw_mutable_data(DataType::make<T>()));
  }
  void* raw_mutable_data() {
    return raw_mutable_data(data_type());
  }
  void* raw_mutable_data(const DataType& dtype);

  bool is_empty() const { return size_ == 0; }
  bool is_dense() const { return layout_type() == kDense; }
  bool is_coo() const { return layout_type() == kCOO; }
  bool is_csr() const { return layout_type() == kCSR; }
  bool is_scalar() const;
  // The type of default layout is dense and its minor_to_major is [rank-1,
  // rank-2, rank-3, ..., 0]
  virtual bool is_default_layout() const;

 private:
  int64_t size_ = 0;
  int64_t offset_ = 0;
  Shape shape_;
  Storage storage_;
};


} // namespace hice
