#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"
#include "hice/core/expression_util.h"

namespace hice {

enum class MatmulOption: int16_t { 
  NoTrans = 100,
  Trans = 101,
  ConjTrans = 102 
};

constexpr MatmulOption kNoTrans = MatmulOption::NoTrans;
constexpr MatmulOption kTrans = MatmulOption::Trans;
constexpr MatmulOption kConjTrans = MatmulOption::ConjTrans;

// Dispatcher
using matmul_kernel_fn_type = 
       void (*)(const Tensor& tensor_a, const Tensor& tensor_b, 
                Tensor& result, MatmulOption option_a,
                MatmulOption option_b, bool resizable);
HICE_DECLARE_DISPATCHER(matmul_dispatcher, matmul_kernel_fn_type);

// outplace
// conjugate is not supportted for now.
HICE_API Tensor matmul(const Tensor& tensor_a, const Tensor& tensor_b, 
                       MatmulOption option_a = kNoTrans,
                       MatmulOption option_b = kNoTrans);

// inplace
// conjugate is not supportted for now.
HICE_API Tensor& matmul(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result, 
                       MatmulOption option_a = kNoTrans,
                       MatmulOption option_b = kNoTrans);

} // namespace hice
