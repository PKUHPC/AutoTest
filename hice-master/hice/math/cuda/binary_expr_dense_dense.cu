#include "hice/core/expression_util.h"
#include "hice/math/binary_expr.h"
#include "hice/math/cuda/eval_expr_dense.cuh"

namespace hice {

namespace {

template <typename TScalarType>
void eval_add_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a + b;
      });
}

void add_kernel(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In add_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to add_cuda must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES(sc_type, "ADD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_add_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_sub_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a - b;
      });
}

void sub_kernel(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In sub_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to sub_cuda must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES(sc_type, "SUB", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_sub_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_mul_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a * b;
      });
}

void mul_kernel(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In mul_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to mul_cuda must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES(sc_type, "MUL", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_mul_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_div_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a / b;
      });
}

void div_kernel(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In div_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to div_cuda must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES(sc_type, "DIV", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_div_expr<scalar_t>(expr);
  });
}

template <typename TScalarType>
void eval_max_expr(Expression& expr) {
  eval_binary_expr<TScalarType, TScalarType, TScalarType>(
      expr, [] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return ::max(a, b);
      });
}

void max_kernel(const Tensor& tensor1, const Tensor& tensor2, Tensor& result, bool resizable) {
  // std::cout << "In max_dense_dense " << std::endl;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to max_cuda must be equal";
  ScalarType sc_type = sc_type_tensor1;
  HICE_DISPATCH_ALL_TYPES(sc_type, "MAX", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(tensor1, tensor2, result, resizable);
    eval_max_expr<scalar_t>(expr);
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(add_dispatcher, &add_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(sub_dispatcher, &sub_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(mul_dispatcher, &mul_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(div_dispatcher, &div_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(max_dispatcher, &max_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(max_dispatcher, &max_kernel,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

}  // namespace hice
