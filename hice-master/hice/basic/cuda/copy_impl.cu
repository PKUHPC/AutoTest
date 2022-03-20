#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/core/expression_util.h"
#include "hice/device/cuda/context_cuda.h"
#include "hice/math/cuda/eval_expr_dense.cuh"

namespace hice {

namespace {

template <typename TScalarType1, typename TScalarType2>
void copy_gpu2gpu_kernel(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_gpu2gpu_kernel" << std::endl;
  Expression expr = ExpressionUtil::make_unary_expr(src, dst, false);
  eval_unary_expr<TScalarType1, TScalarType2>(
      expr, [] __device__(TScalarType1 a) -> TScalarType2 {
        return static_cast<inter_copy_type_t<TScalarType2>>(a);
  });
}

template <typename TScalarType1>
void copy_gpu2gpu_step_impl(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_gpu2gpu_step_impl" << std::endl;
  ScalarType dst_type = dst.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, dst_type, "copy_gpu2gpu_step_impl", [&]() {
    copy_gpu2gpu_kernel<TScalarType1, scalar_t>(src, dst, non_blocking);
  });
}

void copy_gpu2gpu_impl(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_gpu2gpu_impl " << std::endl;
  HICE_CHECK_EQ(src.size(), dst.size());
  ScalarType src_type = src.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, src_type, "copy_gpu2gpu_impl", [&]() {
    copy_gpu2gpu_step_impl<scalar_t>(src, dst, non_blocking);
  });
}

// copy from cpu to gpu with same datatype
template <typename TScalarType>
void copy_cpu2gpu_kernel(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_cpu2gpu_kernel" << std::endl;
  int64_t size = src.size();
  auto data_ptr_src = src.data<TScalarType>();
  auto data_ptr_dst = dst.mutable_data<TScalarType>();
  CUDAContext cuda_ctx(dst.device());
  HICE_CUDA_CHECK(cudaMemcpyAsync( 
    data_ptr_dst,
    data_ptr_src, 
    size * sizeof(TScalarType),
    cudaMemcpyHostToDevice,
    cuda_ctx.stream()));
  if (!non_blocking) {
    cuda_ctx.synchronize();
  }
}

void copy_cpu2gpu_impl(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_cpu2gpu_impl " << std::endl;
  HICE_CHECK_EQ(src.size(), dst.size());
  ScalarType dst_type = dst.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, dst_type, "copy_cpu2gpu_impl", [&]() {
    if (dst_type == src.scalar_type()) {
      copy_cpu2gpu_kernel<scalar_t>(src, dst, non_blocking);
    } else {
      // datatype1(cpu) => datatype2(cpu)
      // Tensor src_with_dst_dtype = empty(src.dims(), src.options().dtype(dst.data_type()));
      Tensor src_with_dst_dtype = empty(src.dims(), dtype(dst.data_type()).device(src.device_type()));
      copy(src, src_with_dst_dtype);
      // datatype2(cpu) => datatype2(gpu)
      copy_cpu2gpu_kernel<scalar_t>(src_with_dst_dtype, dst, non_blocking);
    }
  });
}

// copy from gpu to cpu with same datatype
template <typename TScalarType>
void copy_gpu2cpu_kernel(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_gpu2cpu_kernel" << std::endl;
  int64_t size = src.size();
  auto data_ptr_src = src.data<TScalarType>();
  auto data_ptr_dst = dst.mutable_data<TScalarType>();
  CUDAContext cuda_ctx(src.device());
  HICE_CUDA_CHECK(cudaMemcpyAsync( 
    data_ptr_dst,
    data_ptr_src, 
    size * sizeof(TScalarType),
    cudaMemcpyDeviceToHost,
    cuda_ctx.stream()));
  if (!non_blocking) {
    cuda_ctx.synchronize();
  }
}

void copy_gpu2cpu_impl(const Tensor &src, Tensor &dst, bool non_blocking) {
  // std::cout << "In copy_gpu2cpu_impl " << std::endl;
  HICE_CHECK_EQ(src.size(), dst.size());
  ScalarType src_type = src.scalar_type();
  HICE_DISPATCH_ALL_TYPES_AND(kBool, src_type, "copy_gpu2cpu_impl", [&]() {
    if (src_type == dst.scalar_type()) {
      copy_gpu2cpu_kernel<scalar_t>(src, dst, non_blocking);
    } else {
      // step1: datatype1(gpu) => datatype1(cpu), 
      // must be blocked, or dirty reading occurs in step2.
      // Tensor src_on_cpu = empty(src.dims(), src.options().device(dst.device_type()));
      Tensor src_on_cpu = empty(src.dims(), dtype(src.data_type()).device(dst.device_type()));
      copy_gpu2cpu_kernel<scalar_t>(src, src_on_cpu, false);
      // step2: datatype1(cpu) => datatype2(cpu)
      copy(src_on_cpu, dst);
    }
  });
}



}  // anonymous namespace

HICE_REGISTER_KERNEL(copy_dispatcher, &copy_gpu2gpu_impl,
                      {kCUDA, kDense},
                      {kCUDA, kDense}
);

HICE_REGISTER_KERNEL(copy_dispatcher, &copy_cpu2gpu_impl,
                      {kCPU, kDense},
                      {kCUDA, kDense}
);

HICE_REGISTER_KERNEL(copy_dispatcher, &copy_gpu2cpu_impl,
                      {kCUDA, kDense},
                      {kCPU, kDense}
);

}  // namespace hice
  