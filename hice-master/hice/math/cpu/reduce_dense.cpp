#include "hice/core/expression_util.h"
#include "hice/math/reduce.h"
#include "hice/math/cpu/eval_expr_dense.h"

namespace hice{

namespace {

void reduce_sum_impl(const Tensor& self, ConstIntArrayRef reduced_dims, 
                     bool keep_dim, Tensor& result, 
                     bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "SUM", [&]() {
    scalar_t init_value = 0;
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(expr, init_value, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; });
  });
}

void reduce_prod_impl(const Tensor& self, ConstIntArrayRef reduced_dims,
                      bool keep_dim, Tensor& result, 
                      bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "PROB", [&]() {
    scalar_t init_value = 1;
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(
        expr, init_value,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; });
  });
}

void reduce_mean_impl(const Tensor& self, ConstIntArrayRef reduced_dims,
                      bool keep_dim, Tensor& result, 
                      bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MEAN", [&]() {
    reduce_sum_impl(self, reduced_dims, keep_dim, result, resizable);
    scalar_t factor = 1;
    auto dims = self.dims();
    for (int i = 0; i < reduced_dims.size(); i++) {
      factor *= dims[reduced_dims[i]];
    }
    int num_items = result.size();
    auto result_data = result.mutable_data<scalar_t>();
    for (int i = 0; i < num_items; i++) {
      result_data[i] /= factor;
    }
  });
}

void reduce_and_impl(const Tensor& self, ConstIntArrayRef reduced_dims, 
                     bool keep_dim, Tensor& result, 
                     bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "AND", [&]() {
    scalar_t init_value = 1;
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(
        expr, init_value,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a && b; });
  });
}

void reduce_or_impl(const Tensor& self, ConstIntArrayRef reduced_dims,
                    bool keep_dim, Tensor& result,
                    bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "OR", [&]() {
    scalar_t init_value = 0;
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(expr, init_value, 
                        [=](scalar_t a, scalar_t b) -> scalar_t { return a || b; });
  });
}

void reduce_min_impl(const Tensor& self, ConstIntArrayRef reduced_dims,
                     bool keep_dim, Tensor& result,
                     bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MIN_VALUES", [&]() {
    scalar_t init_value = std::numeric_limits<scalar_t>::max();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(
        expr, init_value, 
        [=](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); });
  });
}

void reduce_max_impl(const Tensor& self, ConstIntArrayRef reduced_dims,
                     bool keep_dim, Tensor& result,
                     bool resizable) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "MAX_VALUES", [&]() {
    scalar_t init_value = std::numeric_limits<scalar_t>::min();
    Expression expr =
        ExpressionUtil::make_reduction_expr(self, result, reduced_dims, 
                                        keep_dim, resizable);
    eval_reduce_expr<scalar_t>(
        expr, init_value, 
        [=](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); });
  });
}
}

// anonymous namespace

HICE_REGISTER_KERNEL(reduce_sum_dispatcher, &reduce_sum_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_prod_dispatcher, &reduce_prod_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_mean_dispatcher, &reduce_mean_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_and_dispatcher, &reduce_and_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_or_dispatcher, &reduce_or_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_min_dispatcher, &reduce_min_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);

HICE_REGISTER_KERNEL(reduce_max_dispatcher, &reduce_max_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense}   // result
);
}  // namespace hice