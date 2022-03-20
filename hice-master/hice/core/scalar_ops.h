#pragma once

#include "hice/core/tensor.h"
#include "hice/basic/factories.h"

namespace hice{

// convert a scalar into tensor
inline Tensor scalar_to_tensor(Scalar a,
                               ScalarType scalar_type,
                               DeviceType device_type) {
  return full({}, a, device(device_type).dtype(scalar_type));
}

}
