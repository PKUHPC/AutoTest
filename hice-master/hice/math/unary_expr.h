#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"

namespace hice {

// Dispatcher
using unary_expr_kernel_fn_type = 
      void (*)(const Tensor& tensor, Tensor& result, bool resizable);
HICE_DECLARE_DISPATCHER(exp_dispatcher, unary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(log_dispatcher, unary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(neg_dispatcher, unary_expr_kernel_fn_type);

// outplace
HICE_API Tensor exp(const Tensor& tensor);
HICE_API Tensor log(const Tensor& tensor);
HICE_API Tensor neg(const Tensor& tensor);

// inplace
HICE_API Tensor& exp(const Tensor& tensor, Tensor& result);
HICE_API Tensor& log(const Tensor& tensor, Tensor& result);
HICE_API Tensor& neg(const Tensor& tensor, Tensor& result);

} // namesapce hice
