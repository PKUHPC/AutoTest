#pragma once

#include <cstring> 
#include "hice/core/tensor.h"
#ifdef HICE_USE_CUDA
#include "hice/device/cuda/common_cuda.h"
#endif

namespace hice {

HICE_API inline void memset(void* data, int value, std::size_t num_bytes,
                     Device device) {
  switch (device.type()) {
    case DeviceType::CPU:
      std::memset(data, value, num_bytes);
      break;
#ifdef HICE_USE_CUDA
    case DeviceType::CUDA:
      cudaMemset(data, value, num_bytes);
      break;
#endif
    default:
      HICE_LOG(ERROR) << "Unsupported device type: " << device;
  }
}

} // namespace hice