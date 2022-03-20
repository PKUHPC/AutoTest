#pragma once

#include "hice/core/allocator.h"

namespace hice {

class CPUAllocator : public Allocator {
 public:
  CPUAllocator() {}

  ~CPUAllocator() override {}

  void* allocate_raw(size_t num_bytes) const override;

  DeleterFnPtr raw_deleter() const override;

};

Allocator* cpu_allocator();

} // namespace hice