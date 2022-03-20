#include "hice/basic/copy.h"
#include "hice/core/expression_util.h"
#include "hice/math/cpu/openmp/parallel.h"
#include "hice/math/cpu/eval_expr_dense.h"

namespace hice {

namespace {

template <typename TScalarType1, typename TScalarType2>
void copy_cpu2cpu_kernel(const Tensor &src, Tensor &dst) {
  // std::cout << "In copy_cpu2cpu_kernel" << std::endl;
  Expression expr = ExpressionUtil::make_unary_expr(src, dst, false);
  eval_unary_expr<TScalarType1, TScalarType2>(
    expr, [](TScalarType1 a) -> TScalarType2 {
       return static_cast<inter_copy_type_t<TScalarType2>>(a); 
    }
  );
}

template <typename TScalarType1>
void copy_cpu2cpu_step_impl(const Tensor &src, Tensor &dst) {
  // std::cout << "In copy_cpu2cpu_step_impl" << std::endl;
  ScalarType dst_type = dst.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, dst_type, "copy_cpu2cpu_step_impl", [&]() {
    copy_cpu2cpu_kernel<TScalarType1, scalar_t>(src, dst);
  });
}

void copy_cpu2cpu_impl(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_cpu2cpu_impl " << std::endl;

  // if (src.is_sparse() && dst.is_sparse()) {
  //   HICE_LOG(ERROR) << "copy() between sparse and sparse Tensors is not implemented!";
  // } else if (src.is_dense() && src.is_dense()) {

  // }
  // HICE_LOG(ERROR) << "copy() between dense and sparse Tensors is not implemented!";

  HICE_CHECK_EQ(src.size(), dst.size());
  ScalarType src_type = src.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, src_type, "copy_cpu2cpu_impl", [&]() {
    copy_cpu2cpu_step_impl<scalar_t>(src, dst);
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(copy_dispatcher, &copy_cpu2cpu_impl,
                     {kCPU, kDense},
                     {kCPU, kDense}
);

}  // namespace hice
