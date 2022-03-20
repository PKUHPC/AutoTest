#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"
#include "hice/core/scalar_ops.h"

namespace hice {

// Dispatcher
using binary_expr_kernel_fn_type = 
      void (*)(const Tensor& tensor1, const Tensor& tensor2, 
               Tensor& result, bool resizable);
HICE_DECLARE_DISPATCHER(add_dispatcher, binary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(sub_dispatcher, binary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(mul_dispatcher, binary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(div_dispatcher, binary_expr_kernel_fn_type);
HICE_DECLARE_DISPATCHER(max_dispatcher, binary_expr_kernel_fn_type);

/* ====outplace==== */
HICE_API Tensor add(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor sub(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor mul(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor div(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor max(const Tensor& tensor1, const Tensor& tensor2);

HICE_API Tensor add(const Tensor& tensor1, Scalar a);
HICE_API Tensor sub(const Tensor& tensor1, Scalar a);
HICE_API Tensor mul(const Tensor& tensor1, Scalar a);
HICE_API Tensor div(const Tensor& tensor1, Scalar a);
HICE_API Tensor max(const Tensor& tensor1, Scalar a);

HICE_API Tensor add(Scalar a, const Tensor& tensor2);
HICE_API Tensor sub(Scalar a, const Tensor& tensor2);
HICE_API Tensor mul(Scalar a, const Tensor& tensor2);
HICE_API Tensor div(Scalar a, const Tensor& tensor2);
HICE_API Tensor max(Scalar a, const Tensor& tensor2);

/* ====inplace==== */
HICE_API Tensor& add(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& sub(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& mul(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& div(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& max(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);

HICE_API Tensor& add(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& sub(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& mul(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& div(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& max(const Tensor& tensor1, Scalar a, Tensor& result);

HICE_API Tensor& add(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& sub(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& mul(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& div(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& max(Scalar a, const Tensor& tensor2, Tensor& result);
} // namesapce hice
