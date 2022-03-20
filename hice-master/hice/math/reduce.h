#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

namespace hice {

// Helper Function for reduce
// Generates a range {0, 1, ..., ndim - 1}
std::vector<int64_t> get_all_dims_range(ConstIntArrayRef dims);

// Dispatcher
using reduction_fn_type = void (*)(const Tensor &self, 
                                   ConstIntArrayRef dim, 
                                   bool keep_dim,
                                   Tensor &result,
                                   bool output_resizable);

HICE_DECLARE_DISPATCHER(reduce_sum_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_prod_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_mean_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_and_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_or_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_min_dispatcher, reduction_fn_type);
HICE_DECLARE_DISPATCHER(reduce_max_dispatcher, reduction_fn_type);

// Operators
HICE_API Tensor reduce_sum(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_prod(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_mean(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_and(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_or(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_min(const Tensor &self, ConstIntArrayRef dim, bool keepdim);
HICE_API Tensor reduce_max(const Tensor &self, ConstIntArrayRef dim, bool keepdim);

// Operators
HICE_API Tensor& reduce_sum(const Tensor &self, ConstIntArrayRef dim, bool keepdim, Tensor &output);
HICE_API Tensor& reduce_mean(const Tensor &self, ConstIntArrayRef dim, bool keepdim, Tensor &output);

}  // namespace hice