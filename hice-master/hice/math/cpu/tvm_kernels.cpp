#if 0
#include "hice/basic/cpu/index_helper.h"
#include "hice/basic/reshape.h"
#include "hice/basic/transpose.h"
#include "hice/core/shape_util.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/matmul.h"

#include "hice/tvm/tvm.h"
#include "hice/tvm/tvm_module.h"
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include <tvm/driver/driver_api.h>
#include <tvm/tir/lowered_func.h>

namespace hice {

namespace {

void matmul_impl(const Tensor &tensor1_, const Tensor &tensor2_, Tensor &result,
                 MatmulOption option_a, MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_tvm_dense_dense" << std::endl;
  Tensor tensor1 = contiguous(tensor1_);
  Tensor tensor2 = contiguous(tensor2_);
  auto ndim_tensor1 = tensor1.ndim();
  auto ndim_tensor2 = tensor2.ndim();
  HICE_CHECK_ARGUMENT(ndim_tensor1 > 0 && ndim_tensor2 > 0)
      << "Both arguments to matmul need to be at least 1D, but they are "
      << ndim_tensor1 << "D and " << ndim_tensor2 << "D";
  int64_t dim_m = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
  int64_t dim_n = option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
  std::vector<int64_t> dims_result = {dim_m, dim_n};
  ExpressionUtil::may_resize_result(result, dims_result, resizable);

  PackedFunc func = TVMHandle::get("sgemm");
  
  runtime::NDArray A = hice::HICETensor_to_NDArray(tensor1);
  runtime::NDArray B = hice::HICETensor_to_NDArray(tensor2);
  runtime::NDArray C = hice::HICETensor_to_NDArray(result);

  auto pa = (float*)A.ToDLPack()->dl_tensor.data;
  auto pb = (float*)B.ToDLPack()->dl_tensor.data;
  auto pc = (float*)C.ToDLPack()->dl_tensor.data;
  func(A, B, C);
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl,
                     {kCPU, kDense},  // first operand
                     {kCPU, kDense},  // second operand
                     {kCPU, kDense}   // result
);

}  // namespace hice

#endif
