#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"
#include "hice/core/scalar_ops.h"

namespace hice {                  

// Dispatcher
using compare_kernel_fn_type = 
      void (*)(const Tensor& tensor1, const Tensor& tensor2, 
               Tensor& result, bool resizable);
HICE_DECLARE_DISPATCHER(equal_dispatcher,         compare_kernel_fn_type);
HICE_DECLARE_DISPATCHER(less_dispatcher,          compare_kernel_fn_type);
HICE_DECLARE_DISPATCHER(less_equal_dispatcher,    compare_kernel_fn_type);
HICE_DECLARE_DISPATCHER(greater_dispatcher,       compare_kernel_fn_type);
HICE_DECLARE_DISPATCHER(greater_equal_dispatcher, compare_kernel_fn_type);

/* ====outplace==== */
HICE_API Tensor equal(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor less(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor less_equal(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor greater(const Tensor& tensor1, const Tensor& tensor2);
HICE_API Tensor greater_equal(const Tensor& tensor1, const Tensor& tensor2);

HICE_API Tensor equal(const Tensor& tensor1, Scalar a);
HICE_API Tensor less(const Tensor& tensor1, Scalar a);
HICE_API Tensor less_equal(const Tensor& tensor1, Scalar a);
HICE_API Tensor greater(const Tensor& tensor1, Scalar a);
HICE_API Tensor greater_equal(const Tensor& tensor1, Scalar a);

HICE_API Tensor equal(Scalar a, const Tensor& tensor2);
HICE_API Tensor less(Scalar a, const Tensor& tensor2);
HICE_API Tensor less_equal(Scalar a, const Tensor& tensor2);
HICE_API Tensor greater(Scalar a, const Tensor& tensor2);
HICE_API Tensor greater_equal(Scalar a, const Tensor& tensor2);

/* ====inplace==== */
HICE_API Tensor& equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& less(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& less_equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& greater(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& greater_equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);

HICE_API Tensor& equal(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& less(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& less_equal(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& greater(const Tensor& tensor1, Scalar a, Tensor& result);
HICE_API Tensor& greater_equal(const Tensor& tensor1, Scalar a, Tensor& result);

HICE_API Tensor& equal(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& less(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& less_equal(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& greater(Scalar a, const Tensor& tensor2, Tensor& result);
HICE_API Tensor& greater_equal(Scalar a, const Tensor& tensor2, Tensor& result);
} // namesapce hice
