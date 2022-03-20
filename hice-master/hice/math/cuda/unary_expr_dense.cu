#include "hice/core/expression_util.h"
#include "hice/math/unary_expr.h"
#include "hice/math/cuda/eval_expr_dense.cuh"

namespace hice{

namespace {

template <typename TScalarType>
void eval_exp_expr(Expression &expr) {
  eval_unary_expr<TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a) -> TScalarType { return ::exp(a); });
}

void exp_impl(const Tensor& tensor, Tensor& result, bool resizable) {
  ScalarType sc_type = tensor.scalar_type();;
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "EXP", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(tensor, result, resizable);
    eval_exp_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_log_expr(Expression &expr) {
  eval_unary_expr<TScalarType, TScalarType>(
      expr, [] __device__ (TScalarType a) -> TScalarType { return ::log(a); });
}

void log_impl(const Tensor& tensor, Tensor& result, bool resizable) {
  ScalarType sc_type = tensor.scalar_type();;
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "LOG", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(tensor, result, resizable);
    eval_log_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_neg_expr(Expression &expr) {
  eval_unary_expr<TScalarType, TScalarType>(
      expr, [] __device__ (TScalarType a) -> TScalarType { return -a; });
}

void neg_impl(const Tensor& tensor, Tensor& result, bool resizable) {
  ScalarType sc_type = tensor.scalar_type();;
  HICE_DISPATCH_ALL_TYPES(sc_type, "NEG", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(tensor, result, resizable);
    eval_neg_expr<scalar_t>(expr);
  });
}

} // anonymous namespace

HICE_REGISTER_KERNEL(
    exp_dispatcher, 
    &exp_impl, 
    {kCUDA, kDense}, // operand
    {kCUDA, kDense} // result 
);

HICE_REGISTER_KERNEL(
    log_dispatcher, 
    &log_impl, 
    {kCUDA, kDense}, // operand
    {kCUDA, kDense} // result 
);

HICE_REGISTER_KERNEL(
    neg_dispatcher, 
    &neg_impl, 
    {kCUDA, kDense}, // operand
    {kCUDA, kDense} // result 
);

} // namespace hice
