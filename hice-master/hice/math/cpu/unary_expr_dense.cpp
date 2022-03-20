#include "hice/core/expression_util.h"
#include "hice/math/unary_expr.h"
#include "hice/math/cpu/eval_expr_dense.h"
#include "hice/math/cpu/vectorization/vec256.h"

namespace hice{

namespace {

template <typename TScalarType>
void eval_exp_expr(Expression &expr) {
  eval_unary_expr_vec<TScalarType, TScalarType>(
      expr, [=](TScalarType a) -> TScalarType { return std::exp(a); },
            [=](Vec256<TScalarType> a) {
              return a.exp();
            });
}

void exp_impl(const Tensor& tensor, Tensor& result, bool resizable) {
  // std::cout << "In exp_impl" << std::endl;
  ScalarType sc_type = tensor.scalar_type();;
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "EXP", [&]() {    
    Expression expr = ExpressionUtil::make_unary_expr(tensor, result, resizable);
    eval_exp_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_log_expr(Expression &expr) {
  eval_unary_expr_vec<TScalarType, TScalarType>(
      expr, [=](TScalarType a) -> TScalarType { return std::log(a); },
            [=](Vec256<TScalarType> a) {
              return a.log();
            });
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
  eval_unary_expr_vec<TScalarType, TScalarType>(
      expr, [=](TScalarType a) -> TScalarType { return -a; },
            [=](Vec256<TScalarType> a) {
              return a.neg();
            });
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
    {kCPU, kDense}, // operand
    {kCPU, kDense} // result 
);

HICE_REGISTER_KERNEL(
    log_dispatcher, 
    &log_impl, 
    {kCPU, kDense}, // operand
    {kCPU, kDense} // result 
);
HICE_REGISTER_KERNEL(
    neg_dispatcher, 
    &neg_impl, 
    {kCPU, kDense}, // operand
    {kCPU, kDense} // result 
);

} // namespace hice
