#include "hice/util/memory.h"
#include "hice/device/cpu/allocator_cpu.h"

namespace hice {

// CPUAllocator implementation
void* CPUAllocator::allocate_raw(size_t num_bytes) const {
  void* p = aligned_malloc(num_bytes, kDefaultAlignBytes);
  return p;
}

DeleterFnPtr CPUAllocator::raw_deleter() const {
  return &aligned_free;
}

//namespace {

static CPUAllocator g_cpu_allocator;
HICE_REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_allocator);

//} // anonymous namespace

Allocator* cpu_allocator() {
  static Allocator* allocator = 
      AllocatorRegistry::singleton()->allocator(DeviceType::CPU);
  return allocator;
}

} // namespace hice

