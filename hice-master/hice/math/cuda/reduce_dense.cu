#include "hice/core/expression_util.h"
#include "hice/math/reduce.h"
#include "hice/math/cuda/eval_expr_dense.cuh"

namespace hice {

namespace {

template <typename TScalarType>
void eval_reduce_sum_expr(Expression &expr, int reduced_dims_size,
                          TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a + b;
      });
}

template <typename TScalarType>
void eval_reduce_prod_expr(Expression &expr, int reduced_dims_size,
                           TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a * b;
      });
}

template <typename TScalarType>
void eval_reduce_mean_expr(Expression &expr, int reduced_dims_size,
                           TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a + b;
      });
}

template <typename TScalarType>
void eval_reduce_and_expr(Expression &expr, int reduced_dims_size,
                          TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a && b;
      });
}

template <typename TScalarType>
void eval_reduce_or_expr(Expression &expr, int reduced_dims_size,
                         TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return a || b;
      });
}

template <typename TScalarType>
void eval_reduce_min_expr(Expression &expr, int reduced_dims_size,
                          TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return (a < b ? a : b);
      });
}

template <typename TScalarType>
void eval_reduce_max_expr(Expression &expr, int reduced_dims_size,
                          TScalarType init_value, TScalarType factor) {
  eval_reduce_expr<TScalarType>(
      expr, reduced_dims_size, init_value, factor,
      [=] __device__(TScalarType a, TScalarType b) -> TScalarType {
        return (a < b ? b : a);
      });
}

void reduce_sum_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                     Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "SUM", [&]() {
    scalar_t init_value = 0;
    scalar_t factor = 1;
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_sum_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_prod_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                      Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "PROD", [&]() {
    scalar_t init_value = 1;
    scalar_t factor = 1;
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_prod_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_mean_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                      Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MEAN", [&]() {
    scalar_t factor = 1;
    for (int i = 0; i < reduced_dims.size(); i++) {
      factor *= self.dims()[reduced_dims[i]];
    }
    scalar_t init_value = 0;
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_mean_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_and_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                     Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "AND", [&]() {
    scalar_t factor = 1;
    scalar_t init_value = 1;
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_and_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_or_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                    Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "OR", [&]() {
    scalar_t factor = 1;
    scalar_t init_value = 0;
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_or_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_min_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                     Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MIN_VALUES", [&]() {
    scalar_t factor = 1;
    scalar_t init_value = std::numeric_limits<scalar_t>::max();
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_min_expr(expr, reduced_dims_size, init_value, factor);
  });
}

void reduce_max_impl(const Tensor &self, ConstIntArrayRef reduced_dims, bool keep_dim,
                     Tensor &result, bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MAX_VALUES", [&]() {
    scalar_t factor = 1;
    scalar_t init_value = std::numeric_limits<scalar_t>::min();
    int reduced_dims_size = reduced_dims.size();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_max_expr(expr, reduced_dims_size, init_value, factor);
  });
}

} // anonymous namespace

HICE_REGISTER_KERNEL(reduce_sum_dispatcher, &reduce_sum_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_prod_dispatcher, &reduce_prod_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_mean_dispatcher, &reduce_mean_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_and_dispatcher, &reduce_and_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_or_dispatcher, &reduce_or_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_min_dispatcher, &reduce_min_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_max_dispatcher, &reduce_max_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense}   // result
);
} // namespace hice
