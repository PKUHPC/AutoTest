#include "hice/core/expression_util.h"
#include "hice/math/cpu/eval_expr_dense.h"
#include "hice/nn/activation.h"

namespace hice {

namespace {

template <typename scalar_t>
void eval_abs_fwd_expr(Expression &expr) {
  eval_unary_expr_vec<scalar_t, scalar_t>(
    expr, [=](scalar_t a) -> scalar_t { return std::abs(a); },
          [=](Vec256<scalar_t> a) {
            return a.abs();
          });
}

void abs_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "ABS_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_abs_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_relu_fwd_expr(Expression &expr) {
  scalar_t zero = 0;
  eval_unary_expr_vec<scalar_t, scalar_t>(
    expr, [=](scalar_t a) -> scalar_t { return std::max(a, zero); },
          [=](Vec256<scalar_t> a) {
            return maximum(a, Vec256<scalar_t>(zero));
          });
}

void relu_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "RELU_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_relu_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_sigmoid_fwd_expr(Expression &expr) {
  scalar_t one = 1;
  scalar_t negative_one = -1;
  eval_unary_expr_vec<scalar_t, scalar_t>(expr, 
    [=](scalar_t a) -> scalar_t {
      return one / (one + std::exp(negative_one * a));
    },
    [=](Vec256<scalar_t> a) {
      a = Vec256<scalar_t>((scalar_t)(0)) - a;
      a = a.exp();
      a = Vec256<scalar_t>((scalar_t)(1)) + a;
      a = a.reciprocal();
      return a;
    });
}

void sigmoid_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "SIGMOID_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_sigmoid_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_sqrt_fwd_expr(Expression &expr) {
  eval_unary_expr_vec<scalar_t, scalar_t>(expr, 
    [=](scalar_t a) -> scalar_t {
      return (scalar_t)std::sqrt(a);
    },
    [=](Vec256<scalar_t> a) {
      return a.sqrt();
    });
}

void sqrt_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "SQRT_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_sqrt_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_square_fwd_expr(Expression &expr) {
  eval_unary_expr_vec<scalar_t, scalar_t>(
    expr, 
    [=](scalar_t a) -> scalar_t { return std::pow(a, 2); },
    [=](Vec256<scalar_t> a) {
      return a.pow(Vec256<scalar_t>(2));
    });
}

void square_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "SQUARE_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_square_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_tanh_fwd_expr(Expression &expr) {
  eval_unary_expr_vec<scalar_t, scalar_t>(
    expr, 
    [=](scalar_t a) -> scalar_t { return std::tanh(a); },
    [=](Vec256<scalar_t> a) {
      return a.tanh();
    });
}

void tanh_fwd_impl(const Tensor& input, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "TANH_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_tanh_fwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_elu_fwd_expr(Expression &expr, const Tensor& alpha_t) {
  scalar_t alpha = alpha_t.data<scalar_t>()[0];
  scalar_t zero = 0;
  scalar_t one = 1;
  eval_unary_expr_vec<scalar_t, scalar_t>(
    expr, 
    [=](scalar_t a) -> scalar_t {
      /// NOTE: Here if a > 0, out = a; if a < 0, out = alpha*(exp(a) - 1);
      scalar_t case_1 = (a < zero) * alpha * (std::exp(a) - one);
      scalar_t case_2 = (a >= zero) * a;
      return case_1 + case_2;
    },
    [=](Vec256<scalar_t> a) {
      auto zero_vec = Vec256<scalar_t>(zero);
      auto one_vec = Vec256<scalar_t>(one);
      auto alpha_vec = Vec256<scalar_t>(alpha);
      auto cmp_vec = Vec256<scalar_t>::blendv(zero_vec, one_vec, a < zero_vec);
      auto case_1_vec = alpha_vec * (a.exp() - one_vec) * cmp_vec;
      auto case_2_vec = (one_vec - cmp_vec) * a;
      return case_1_vec + case_2_vec;
    });
}

void elu_fwd_impl(const Tensor& input, const Tensor& alpha, Tensor& output, bool resizable) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "ELU_FWD", [&]() {
    Expression expr = ExpressionUtil::make_unary_expr(input, output, resizable);
    eval_elu_fwd_expr<scalar_t>(expr, alpha);
  });
}


template <typename scalar_t>
void eval_abs_bwd_expr(Expression &expr) {
  scalar_t zero = 0;
  scalar_t one = 1;
  scalar_t negative_one = -1;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t prim = (in > 0) - (in < 0);
      return grad_out * prim;
    }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto zero_vec = Vec256<scalar_t>(zero);
      auto one_vec = Vec256<scalar_t>(one);
      // Comparision operators returns bitmask.
      auto left = Vec256<scalar_t>::blendv(zero_vec, one_vec, zero_vec < in);
      auto right = Vec256<scalar_t>::blendv(zero_vec, one_vec, in < zero_vec);
      return (left - right) * grad_out;
    });                                                                                                                                                                                                                          
}

void abs_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "ABS_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_abs_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_relu_bwd_expr(Expression &expr) {
  scalar_t one = 1;
  scalar_t zero = 0;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t prim = (in >= 0) - (in == 0);
      return grad_out * prim;
    }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto zero_vec = Vec256<scalar_t>(zero);
      auto one_vec = Vec256<scalar_t>(one);
      // Comparision operators returns bitmask.
      auto left = Vec256<scalar_t>::blendv(zero_vec, one_vec, zero_vec <= in);
      auto right = Vec256<scalar_t>::blendv(zero_vec, one_vec, in == zero_vec);
      return (left - right) * grad_out;
    });  
}

void relu_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "RELU_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_relu_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_sigmoid_bwd_expr(Expression &expr) {
  scalar_t one = 1;
  scalar_t negative_one = -1;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t e = one / (one + std::exp(negative_one * in));
      return grad_out * e * (one - e);
    }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto one_vec = Vec256<scalar_t>(one);
      auto e_vec = one_vec / (one_vec + in.neg().exp());
      return grad_out * e_vec * (one_vec - e_vec);
    }); 
}

void sigmoid_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "SIGMOID_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_sigmoid_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_sqrt_bwd_expr(Expression &expr) {
  scalar_t one = 1;
  scalar_t two = 2;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t prim = one / (two * std::sqrt(in));
      return prim * grad_out;
    }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto two_vec = Vec256<scalar_t>(two);
      return in.rsqrt() / two_vec * grad_out;
    }); 
}

void sqrt_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "SQRT_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_sqrt_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_square_bwd_expr(Expression &expr) {
  scalar_t two = 2;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(
    expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t { return grad_out * two * in; }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto two_vec = Vec256<scalar_t>(two);
      return in * two_vec * grad_out;
    }); 
}

void square_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "SQUARE_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_square_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_tanh_bwd_expr(Expression &expr) {
  scalar_t one = 1;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(
    expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t e = std::tanh(in);
      return grad_out * (one - e) * (one + e); 
    }, 
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto e_vec = in.tanh();
      auto one_vec = Vec256<scalar_t>(one);
      return grad_out * (one_vec - e_vec) * (one_vec + e_vec); 
    }); 
}

void tanh_bwd_impl(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "TANH_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_tanh_bwd_expr<scalar_t>(expr);
  });
}

template <typename scalar_t>
void eval_elu_bwd_expr(Expression &expr, const Tensor& alpha_t) {
  scalar_t alpha = alpha_t.data<scalar_t>()[0];
  scalar_t zero = 0;
  scalar_t one = 1;
  eval_binary_expr_vec<scalar_t, scalar_t, scalar_t>(
    expr, 
    [=](scalar_t in, scalar_t grad_out) -> scalar_t {
      scalar_t case_1 = (in < zero) * alpha * std::exp(in);
      scalar_t case_2 = (in >= zero);
      return (case_1 + case_2) * grad_out;
    },
    [=](Vec256<scalar_t> in, Vec256<scalar_t> grad_out) {
      auto zero_vec = Vec256<scalar_t>(zero);
      auto one_vec = Vec256<scalar_t>(one);
      auto alpha_vec = Vec256<scalar_t>(alpha);
      auto cmp_vec = Vec256<scalar_t>::blendv(zero_vec, one_vec, in < zero_vec);
      auto case_1_vec = cmp_vec * (alpha_vec * in.exp());
      auto case_2_vec = one_vec - cmp_vec;
      return (case_1_vec + case_2_vec) * grad_out;
    });
}

void elu_bwd_impl(const Tensor& input, const Tensor& alpha, 
                  const Tensor& grad_output, Tensor& grad_input, bool resizable) {
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_grad_output = grad_output.scalar_type();
  HICE_CHECK(sc_type_input == sc_type_grad_output);
  HICE_DISPATCH_FLOATING_TYPES(sc_type_input, "ELU_BWD", [&]() {
    Expression expr = ExpressionUtil::make_binary_expr(input, grad_output, grad_input, resizable);
    eval_elu_bwd_expr<scalar_t>(expr, alpha);
  });
}

}  // namespace

// Forward
HICE_REGISTER_KERNEL(abs_fwd_dispatcher, &abs_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(relu_fwd_dispatcher, &relu_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(sigmoid_fwd_dispatcher, &sigmoid_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(sqrt_fwd_dispatcher, &sqrt_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(square_fwd_dispatcher, &square_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(tanh_fwd_dispatcher, &tanh_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(elu_fwd_dispatcher, &elu_fwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);

// Backward
HICE_REGISTER_KERNEL(sqrt_bwd_dispatcher, &sqrt_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(abs_bwd_dispatcher, &abs_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(relu_bwd_dispatcher, &relu_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(sigmoid_bwd_dispatcher, &sigmoid_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(square_bwd_dispatcher, &square_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(tanh_bwd_dispatcher, &tanh_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);
HICE_REGISTER_KERNEL(elu_bwd_dispatcher, &elu_bwd_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense},  // third operand
                     {kCPU, kDense}   // result
);

}  // namespace hice
