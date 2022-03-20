#ifdef HICE_USE_MKL
// #if 1

#include "hice/basic/transpose.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/cpu/matmul_sparse_util_mkl.h"
#include "hice/math/matmul.h"
#include "hice/device/cpu/common_mkl.h"

namespace hice {

namespace {

// sparse = matmul(sparse, sparse)
void spgemm(const Tensor &tensor1, const Tensor &tensor2, Tensor &result,
          MatmulOption option_a, MatmulOption option_b) {
  // check scalar type.
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to mm must be equal";
  // check dims.
  const int n_rows_tensor1 = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
  const int n_cols_tensor1 = option_a == kNoTrans ? tensor1.dim(1) : tensor1.dim(0);
  const int n_rows_tensor2 = option_b == kNoTrans ? tensor2.dim(0) : tensor2.dim(1);
  const int n_cols_tensor2 = option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
  HICE_CHECK_EQ(n_cols_tensor1, n_rows_tensor2)
      << "Dimensions of arguments to mm must be matched";

  sparse_operation_t trans_a = trans_option_from_(option_a);
  sparse_operation_t trans_b = trans_option_from_(option_b);

  sparse_matrix_t csrA;
  sparse_matrix_t csrB;
  sparse_matrix_t csrC;
  csr_from_(tensor1, &csrA, trans_a);
  csr_from_(tensor2, &csrB, trans_b);
  HICE_MKLSPARSE_CHECK(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC));
  store_csr_into_(csrC, result);
  result = result.to_coalesced();
  HICE_MKLSPARSE_CHECK(mkl_sparse_destroy(csrA));
  HICE_MKLSPARSE_CHECK(mkl_sparse_destroy(csrB));
  HICE_MKLSPARSE_CHECK(mkl_sparse_destroy(csrC));
}

// sparse = matmul(sparse, sparse)
void matmul_spgemm_impl(const Tensor &tensor1, const Tensor &tensor2,
                      Tensor &result, MatmulOption option_a,
                      MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_cpu_sparse_sparse_sparse" << std::endl;
  const int64_t ndim_tensor1 = tensor1.ndim();
  const int64_t ndim_tensor2 = tensor2.ndim();
  HICE_CHECK(ndim_tensor1 == 2 && ndim_tensor2 == 2)
      << "For sparse-sparse matmul, Only mm is supported, but the inputs are "
      << ndim_tensor1 << "D and " << ndim_tensor2 << "D";
  HICE_CHECK(resizable);
  int64_t dim_m =
      option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
  int64_t dim_n =
      option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
    std::vector<int64_t> dims_result = {dim_m, dim_n};
  result.resize(dims_result);
  spgemm(tensor1, tensor2, result, option_a, option_b);
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_spgemm_impl,
                     {kCPU, kCOO},  // first operand
                     {kCPU, kCOO},  // second operand
                     {kCPU, kCOO}   // result
);

}  // namespace hice

#endif