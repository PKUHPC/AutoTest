#pragma once

#include <map>
#include <mutex> 

#include "hice/util/copy_bytes.h"
#include "hice/util/type_id.h"
#include "hice/util/registry.h"
#include "hice/core/device.h"

namespace hice {

class DeviceContext {
 public:
  virtual ~DeviceContext() = default; 

  virtual Device device() const = 0;

  virtual DeviceType device_type() const = 0;

  virtual void switch_to_device(int /*stream_id*/) = 0;

  virtual void synchronize() = 0;

  void copy_bytes(size_t nbytes, 
                  const void* src, 
                  Device src_device,
                  void* dst, 
                  Device dst_device,
                  bool async = true) {
    hice::copy_bytes(nbytes, src, src_device, dst, dst_device, async);
  }

  void copy_items(size_t nitems, 
                  DataType dtype, 
                  const void* src, 
                  Device src_device,
                  void* dst, 
                  Device dst_device,
                  bool async = true) {
    hice::copy_bytes(nitems * dtype.size(), src, src_device, dst, dst_device, async);
  }

};

// Context constructor registry
HICE_DECLARE_TYPED_REGISTRY(
    ContextRegistry,
    DeviceType,
    DeviceContext,
    std::unique_ptr,
    Device);

#define HICE_REGISTER_CONTEXT(key, ...)   \
  HICE_REGISTER_TYPED_CLASS(ContextRegistry, key, __VA_ARGS__)

inline std::unique_ptr<DeviceContext> create_context(const Device& device) {
  return ContextRegistry()->create(device.type(), device);
}
  
} // namespace hice