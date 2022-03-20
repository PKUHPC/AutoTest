#ifdef HICE_USE_MKL
// #if 1

#include "hice/basic/transpose.h"
#include "hice/basic/reshape.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/matmul.h"
#include "hice/math/cpu/matmul_sparse_util_mkl.h"
#include "hice/device/cpu/common_mkl.h"

namespace hice {

namespace {

// matmul between matrix and vector
void spmv(const Tensor &tensor1, const Tensor &tensor2, Tensor &result,
        MatmulOption option) {
  // check scalar type.
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to mv must be equal";
  // check dims
  const int n_rows = option == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
  const int n_cols = option == kNoTrans ? tensor1.dim(1) : tensor1.dim(0);
  HICE_CHECK_EQ(n_cols, tensor2.dim(0))
      << "Dimensions of arguments to mv must be matched";

  sparse_matrix_t csrA;
  mklcsr_from_csr(tensor1, &csrA);
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_operation_t trans = trans_option_from_(option);
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      HICE_MKLSPARSE_CHECK(mkl_sparse_s_mv(trans, alpha, csrA, descrA, x, beta, y));
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      HICE_MKLSPARSE_CHECK(mkl_sparse_d_mv(trans, alpha, csrA, descrA, x, beta, y));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  mkl_sparse_destroy(csrA);
}

// matmul between sparse_matrix and dense_matrix
void spmm(const Tensor &tensor1, const Tensor &tensor2, Tensor &result,
        MatmulOption option_a) {
  // check scalar type.
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to mm must be equal";
  // check dims.
  const int n_rows_tensor1 = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
  const int n_cols_tensor1 = option_a == kNoTrans ? tensor1.dim(1) : tensor1.dim(0);
  const int n_rows_tensor2 = tensor2.dim(0);
  const int n_cols_tensor2 = tensor2.dim(1);
  HICE_CHECK_EQ(n_cols_tensor1, n_rows_tensor2)
      << "Dimensions of arguments to mm must be matched";
  sparse_matrix_t csrA;
  mklcsr_from_csr(tensor1, &csrA);
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_operation_t trans_a = trans_option_from_(option_a);
  const int ldx = n_cols_tensor2;
  const int n_cols_result = result.dim(1);
  const int ldy = n_cols_result;
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      HICE_MKLSPARSE_CHECK(mkl_sparse_s_mm(trans_a, alpha, csrA, descrA, SPARSE_LAYOUT_ROW_MAJOR, x,
                      n_cols_result, ldx, beta, y, ldy));
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      HICE_MKLSPARSE_CHECK(mkl_sparse_d_mm(trans_a, alpha, csrA, descrA, SPARSE_LAYOUT_ROW_MAJOR, x,
                      n_cols_result, ldx, beta, y, ldy));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  HICE_MKLSPARSE_CHECK(mkl_sparse_destroy(csrA));
}

void matmul_impl(const Tensor &tensor1, const Tensor &tensor2, Tensor &result,
                 MatmulOption option_a, MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_cpu_csr_dense" << std::endl;
  auto ndim_tensor1 = tensor1.ndim();
  auto ndim_tensor2 = tensor2.ndim();
  if (ndim_tensor1 == 2 && ndim_tensor2 == 1) {
    int64_t dim_m = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
    std::vector<int64_t> dims_result = {dim_m};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    spmv(tensor1, tensor2, result, option_a);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 2) {
    int64_t dim_m = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
    int64_t dim_n = option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
    std::vector<int64_t> dims_result = {dim_m, dim_n};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    Tensor tensor2_new =
        option_b == kTrans ? contiguous(transpose_matrix(tensor2)) : tensor2;
    spmm(tensor1, tensor2_new, result, option_a);
  } else {
    HICE_CHECK(false) << "For sparse-dense matmul, Only mv and mm is "
                         "supported, but the inputs are "
                      << ndim_tensor1 << "D and " << ndim_tensor2 << "D";
  }
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl,
                     {kCPU, kCSR},  // first operand
                     {kCPU, kDense},   // second operand
                     {kCPU, kDense}    // result
);

}  // namespace hice

#endif
