#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

namespace hice {

// Dispatcher
using arg_reduce_fn_type = void (*)(const Tensor &self, int64_t dim,
                                Tensor &result, Tensor &result_indices);

HICE_DECLARE_DISPATCHER(max_tuple_dispatcher, arg_reduce_fn_type);
HICE_DECLARE_DISPATCHER(min_tuple_dispatcher, arg_reduce_fn_type);

// Operators
HICE_API std::tuple<Tensor, Tensor> min(Tensor &self, int64_t dim, bool keepdim);
HICE_API std::tuple<Tensor, Tensor> max(Tensor &self, int64_t dim, bool keepdim);
HICE_API Tensor argmin(Tensor &self, int64_t dim, bool keepdim);
HICE_API Tensor argmax(Tensor &self, int64_t dim, bool keepdim);

}  // namespace hice