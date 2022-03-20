#include "hice/core/expression_util.h"
#include "hice/math/compare.h"
#include "hice/math/cpu/eval_expr_dense.h"

namespace hice {

namespace {

template <typename TScalarType>
void eval_equal_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, bool>(
      expr, [](TScalarType a, TScalarType b) -> bool { return a == b; });
}

void equal_impl(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In equal_impl " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to equal_cpu must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "EQUAL", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_equal_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_less_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, bool>(
      expr, [](TScalarType a, TScalarType b) -> bool { return a < b; });
}

void less_impl(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In less_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to less_cpu must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "LESS", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_less_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_less_equal_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, bool>(
      expr, [](TScalarType a, TScalarType b) -> bool { return a <= b; });
}

void less_equal_impl(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In less_equal_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to less_equal_cpu must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "LESS_EQUAL", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_less_equal_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_greater_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, bool>(
      expr, [](TScalarType a, TScalarType b) -> bool { return a > b; });
}

void greater_impl(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In greater_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to greater_cpu must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "GREATER", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_greater_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_greater_equal_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, bool>(
      expr, [](TScalarType a, TScalarType b) -> bool { return a >= b; });
}

void greater_equal_impl(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In greater_equal_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to greater_equal_cpu must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "GREATER_EQUAL", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_greater_equal_expr<scalar_t>(expr);
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(equal_dispatcher, &equal_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(less_dispatcher, &less_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(less_equal_dispatcher, &less_equal_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(greater_dispatcher, &greater_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(greater_equal_dispatcher, &greater_equal_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);

}  // namespace hice
