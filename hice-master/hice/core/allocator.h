#pragma once

#include <mutex>
#include <cstddef>
#include <unordered_map>

#include "hice/util/loguru.h"
#include "hice/util/data_ptr.h"
#include "hice/core/device.h"

namespace hice {

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual void* allocate_raw(size_t num_bytes) const = 0;

  void deallocate_raw(void* ptr) const {
    auto deleter = raw_deleter();
    deleter(ptr);
  }

  virtual DeleterFnPtr raw_deleter() const = 0;

  DataPtr allocate(size_t num_bytes) const {
    void *raw_ptr = allocate_raw(num_bytes);
    auto deleter = raw_deleter();
    return {raw_ptr, raw_ptr, deleter};
  }

};

class AllocatorRegistry {
 public:
   static AllocatorRegistry* singleton() {
     static AllocatorRegistry *alloc_registry = new AllocatorRegistry();
     return alloc_registry;
   }

   void set_allocator(DeviceType device_type, Allocator *allocator) {
     std::lock_guard<std::mutex> guard(mu_);
     //std::cout << "set_alloc" << static_cast<int>(device_type) << std::endl;
     allocators_[static_cast<int>(device_type)] = allocator;
   } 

   Allocator* allocator(DeviceType device_type) {
     //std::cout << "alloc" << static_cast<int>(device_type) << std::endl;
     return allocators_[static_cast<int>(device_type)];
   }

 private:
   AllocatorRegistry() {}  
   AllocatorRegistry(AllocatorRegistry const&) = delete;
   AllocatorRegistry& operator=(AllocatorRegistry const&) = delete;
   std::mutex mu_;
   std::unordered_map<int, Allocator*> allocators_;
   //Allocator* allocators_[kNumDeviceTypes];
};

struct AllocatorRegister {
  AllocatorRegister(DeviceType device_type, Allocator *allocator) {
    //std::cout << "alloc_register" << static_cast<int>(device_type) << std::endl;
    AllocatorRegistry::singleton()->set_allocator(device_type, allocator);
  }
};

#define HICE_REGISTER_ALLOCATOR(device, allocator)                        \
  static AllocatorRegister g_allocator_register(device, allocator)        \

Allocator* get_allocator(DeviceType device_type);

} // namespace hice