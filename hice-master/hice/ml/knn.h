#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"

namespace hice {

// Dispatcher
using knn_kernel_fn_type = void (*)(const Tensor& ref, const Tensor& labels, const Tensor& query, int k, Tensor& result);
HICE_DECLARE_DISPATCHER(knn_dispatcher, knn_kernel_fn_type);

// Operators
HICE_API Tensor knn(const Tensor& ref, const Tensor& labels, const Tensor& query, int k);

} // namesapce hice
