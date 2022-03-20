#pragma once

#include <hice/basic/factories.h>
#include <hice/core/tensor.h>
#include <hice/util/dlpack.h>

namespace hice {

HICE_API DLTensor HICETensor_to_DLTensor(const hice::Tensor& self_const);

HICE_API DLManagedTensor HICETensor_to_DLManagedTensor(const hice::Tensor& self_const);

// NOTE: strides and byte_offset are ignored
HICE_API Tensor DLTensor_to_HICETensor(const DLTensor& self_const);

} // namespace hice