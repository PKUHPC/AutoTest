#pragma once

#include "hice/util/type_id.h"
#include "hice/util/data_ptr.h"
#include "hice/util/intrusive_ptr.h"
#include "hice/core/device.h"
#include "hice/core/dimension.h"
#include "hice/core/allocator.h"
#include "hice/core/scalar_type.h"

namespace hice {

class HICE_API StorageImpl : public intrusive_ptr_target {
 public:
  StorageImpl(size_t size, DataPtr &&data_ptr, DataType dtype, Device device,
              Allocator* allocator)
      : size_(size),
        data_ptr_(std::move(data_ptr)),
        dtype_(dtype),
        device_(device),
        allocator_(allocator) {}

  StorageImpl(size_t size, DataType dtype, Device device, Allocator* allocator)
      : StorageImpl(size, allocator->allocate(size * dtype.size()), dtype,
                    device, allocator) {}

  StorageImpl() = delete;

  StorageImpl(StorageImpl&& other) = default;

  StorageImpl(const StorageImpl&) = delete;

  StorageImpl& operator=(StorageImpl&& other) = default;

  StorageImpl& operator=(const StorageImpl&) = delete;
  
  size_t size() const { return size_; };

  void set_size(size_t size) { size_ = size; }

  const DataType& data_type() const { return dtype_; }

  void set_data_type(const DataType& dtype) { dtype_ = dtype; }

  Device device() const { return device_; }

  Allocator* allocator() const { return allocator_; }

  template <typename T> bool is_type() const { return dtype_.match<T>(); }

  template <typename T>
  T* data() const { return static_cast<T*>(raw_data()); }

  template <typename T>
  T* data() { return static_cast<T*>(raw_data()); }

  void* raw_data() const { return data_ptr_.get(); }

  void* raw_data() { return data_ptr_.get(); }

  const DataPtr& data_ptr() const { return data_ptr_; }

  DataPtr& data_ptr() { return data_ptr_; }

  DataPtr set_data_ptr(DataPtr&& data_ptr) {
    std::swap(data_ptr_, data_ptr);
    return std::move(data_ptr);
  }

 private:
  size_t size_;
  DataPtr data_ptr_;
  DataType dtype_;
  Device device_;
  Allocator *allocator_;
};

} // namespace hice
