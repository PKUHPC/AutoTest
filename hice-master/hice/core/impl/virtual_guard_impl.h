// This file is based on c10\core\impl\VirtualGuardImpl.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 
#pragma once

#include "hice/core/impl/device_guard_impl_interface.h"

namespace hice {
namespace impl {

/**
 * An implementation of DeviceGuardImplInterface which delegates
 * to virtual dispatch on the DeviceGuardImpl registry.
 */
class VirtualGuardImpl final : public DeviceGuardImplInterface {
public:
  VirtualGuardImpl(DeviceType device_type)
    : impl_(get_device_guard_impl(device_type)) {}
  // This constructor exists purely for testing
  VirtualGuardImpl(const DeviceGuardImplInterface* impl)
    : impl_(impl) {}

  // Copying and moving is OK!

  DeviceType type() const override {
    return impl_->type();
  }
  Device exchange_device(Device d) const override {
    return impl_->exchange_device(d);
  }
  Device get_device() const override {
    return impl_->get_device();
  }
  void set_device(Device d) const override {
    impl_->set_device(d);
  }
  void unchecked_set_device(Device d) const noexcept override {
    impl_->unchecked_set_device(d);
  }
  Stream get_stream(Device d) const noexcept override {
    return impl_->get_stream(d);
  }
  Stream exchange_stream(Stream s) const noexcept override {
    return impl_->exchange_stream(s);
  }
  DeviceIndex device_count() const noexcept override {
    return impl_->device_count();
  }
private:
  const DeviceGuardImplInterface* impl_ = nullptr;
};

}} // namespace hice::impl