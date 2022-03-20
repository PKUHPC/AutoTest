#include "hice/core/allocator.h"

namespace hice {

Allocator* get_allocator(DeviceType device_type) {
  Allocator* alloc = AllocatorRegistry::singleton()->allocator(device_type);
  HICE_CHECK_NOTNULL(alloc) <<  "Allocator for " << device_type << " is not set.";
  return alloc;
}


} // namespace hice