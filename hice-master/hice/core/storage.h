#pragma once

#include "hice/util/loguru.h"
#include "hice/util/intrusive_ptr.h"
#include "hice/core/storage_impl.h"

namespace hice {

class HICE_API Storage {
 private:
  using StorageImplPtr = intrusive_ptr<StorageImpl>;
 public:
  Storage() = default;

  Storage(StorageImplPtr ptr) : pimpl_(std::move(ptr)) {}

  Storage(size_t size, DataType dtype, Device device, Allocator* allocator)
      : pimpl_(make_intrusive<StorageImpl>(size, dtype, device, allocator)){};

  Storage(size_t size, DataPtr&& data_ptr, DataType dtype, Device device,
          Allocator* allocator)
      : pimpl_(make_intrusive<StorageImpl>(size, std::move(data_ptr), dtype,
                                           device, allocator)){};

  int64_t size() const { return pimpl_->size(); }

  void set_size(size_t size) { pimpl_->set_size(size); }

  const DataType& data_type() const { return pimpl_->data_type(); }

  void set_data_type(const DataType &dtype) { pimpl_->set_data_type(dtype); }

  Device device() const { return pimpl_->device(); }

  Allocator* allocator() const { return pimpl_->allocator(); }

  template <typename T> bool is_type() const { return pimpl_->is_type<T>(); }

  template <typename T>
  T* data() { return static_cast<T*>(pimpl_->data<T>()); }

  template <typename T>
  T* data() const { return static_cast<T*>(pimpl_->data<T>()); }

  void* raw_data() { return pimpl_->raw_data(); }

  void* raw_data() const { return pimpl_->raw_data(); }

  DataPtr& data_ptr() { return pimpl_->data_ptr(); }

  const DataPtr& data_ptr() const { return pimpl_->data_ptr(); }

  DataPtr set_data_ptr(DataPtr&& data_ptr) {
    return pimpl_->set_data_ptr(std::move(data_ptr));
  }

  operator bool() const { return pimpl_; }
  const StorageImpl& impl() const { return *pimpl_.get(); }
  StorageImpl& mutable_impl() { return *pimpl_.get(); }

 private:
  StorageImplPtr pimpl_;
};

template <typename T, typename... Args>
Storage make_storage(Args&&... args) {
  return Storage(hice::make_intrusive<T>(std::forward<Args>(args)...));
}


} // namespace hice
