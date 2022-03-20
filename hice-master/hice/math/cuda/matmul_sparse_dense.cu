#if 0

#include "hice/basic/transpose.h"
#include "hice/basic/reshape.h"
#include "hice/core/sparse_tensor.h"
#include "hice/device/cuda/context_cuda.h"
#include "hice/device/cuda/allocator_cuda.h"
#include "hice/math/cuda/matmul_sparse_util.cuh"
#include "hice/math/matmul.h"

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
  // cuda handle
  CUDAContext cuda_ctx(tensor1.device());
  cusparseHandle_t handle = cuda_ctx.cusparse_handle();
  HICE_CUSPARSE_CHECK(
      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  // sparse mat desc
  cusparseMatDescr_t descr = 0;
  HICE_CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  HICE_CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  HICE_CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  cusparseOperation_t trans = trans_option_from_(option);
  // convert coo to csr
  const Tensor& values = tensor1.values();
  const Tensor& indices = tensor1.indices();
  const int n_nonzero = tensor1.n_nonzero();
  const int *rowind_coo = indices.data<int>();
  const int *colind = rowind_coo + n_nonzero;
  int *rowind_csr = (int *)cuda_allocator()->allocate_raw((n_rows + 1) * sizeof(int));
  HICE_CUSPARSE_CHECK(cusparseXcoo2csr(handle, rowind_coo, n_nonzero, n_rows,
                                       rowind_csr, CUSPARSE_INDEX_BASE_ZERO));
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      const scalar_t *val_ptr = values.data<scalar_t>();
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      cusparseScsrmv(handle, trans, n_rows, n_cols, n_nonzero, &alpha, descr,
                     val_ptr, rowind_csr, colind, x, &beta, y);
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      const scalar_t *val_ptr = values.data<scalar_t>();
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      cusparseDcsrmv(handle, trans, n_rows, n_cols, n_nonzero, &alpha, descr,
                     val_ptr, rowind_csr, colind, x, &beta, y);
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  cuda_allocator()->deallocate_raw(rowind_csr);
}

// matmul between matrix and matrix
void spmm(const Tensor &tensor1, const Tensor &tensor2, Tensor &result) {
  // check scalar type.
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2)
      << "Both scalar types of arguments to mm must be equal";
  // check dims.
  const int n_rows_tensor1 = tensor1.dim(0);
  const int n_cols_tensor1 = tensor1.dim(1);
  const int n_rows_tensor2 = tensor2.dim(0);
  const int n_cols_tensor2 = tensor2.dim(1);
  HICE_CHECK_EQ(n_cols_tensor1, n_rows_tensor2)
      << "Dimensions of arguments to mm must be matched";
  const int m = n_rows_tensor1;
  const int k = n_cols_tensor1;
  const int n = n_cols_tensor2;
  // cuda handle
  CUDAContext cuda_ctx(tensor1.device());
  cublasHandle_t cublas_handle = cuda_ctx.cublas_handle();
  HICE_CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  cusparseHandle_t cusparse_handle = cuda_ctx.cusparse_handle();
  HICE_CUSPARSE_CHECK(
      cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST));
  // sparse mat desc
  cusparseMatDescr_t descr = 0;
  HICE_CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  HICE_CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  HICE_CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  // convert coo to csr
  int nnzA = tensor1.n_nonzero();
  const Tensor& indicesA = tensor1.indices();
  const int *rowindA_coo = indicesA.data<int>();
  const int *colindA = indicesA.data<int>() + nnzA;
  int *rowindA_csr = NULL;
  cudaMalloc((void **)&rowindA_csr, sizeof(int) * (n_rows_tensor1 + 1));
  HICE_CUSPARSE_CHECK(cusparseXcoo2csr(cusparse_handle, rowindA_coo, nnzA,
                                       n_rows_tensor1, rowindA_csr,
                                       CUSPARSE_INDEX_BASE_ZERO));
  // temp C
  Tensor temp_C(device(tensor1.device()).dtype(tensor1.data_type()));
  temp_C.resize({m, n});
  // kernel
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      const scalar_t *valuesA = tensor1.values().data<scalar_t>();
      const scalar_t *B = tensor2.data<scalar_t>();
      scalar_t *C = result.mutable_data<scalar_t>();
      scalar_t *Ct = temp_C.mutable_data<scalar_t>();
      int ldb = n;
      int ldc = m;
      cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
        nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
        B, ldb, &beta, Ct, ldc);
      // convert Ct to C.
      const int ldc_Ct = m;  // leading dimension of Ct
      const int ldc_C = n;
      scalar_t one = 1.0;
      scalar_t zero = 0.0;
      cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, &one, Ct,
                  ldc_Ct, &zero, Ct, ldc_Ct, C, ldc_C);
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      scalar_t alpha = 1.0;
      scalar_t beta = 0.0;
      const scalar_t *valuesA = tensor1.values().data<scalar_t>();
      const scalar_t *B = tensor2.data<scalar_t>();
      scalar_t *C = result.mutable_data<scalar_t>();
      scalar_t *Ct = temp_C.mutable_data<scalar_t>();
      int ldb = n;
      int ldc = m;
      cusparseDcsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
        nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
        B, ldb, &beta, Ct, ldc);
      // convert Ct to C.
      const int ldc_Ct = m;  // leading dimension of Ct
      const int ldc_C = n;
      scalar_t one = 1.0;
      scalar_t zero = 0.0;
      cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, &one, Ct,
                  ldc_Ct, &zero, Ct, ldc_Ct, C, ldc_C);
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  cudaFree(rowindA_csr);
}

void matmul_impl(const Tensor &tensor1_, const Tensor &tensor2, Tensor &result,
                 MatmulOption option_a, MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_cuda_sparse_dense" << std::endl;
  Tensor tensor1 = tensor1_.is_coalesced() ? tensor1_ : tensor1_.to_coalesced();
  auto ndim_tensor1 = tensor1.ndim();
  auto ndim_tensor2 = tensor2.ndim();
  if (ndim_tensor1 == 2 && ndim_tensor2 == 1) {
    int64_t dim_m = option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
    std::vector<int64_t> dims_result = {dim_m};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    spmv(tensor1, tensor2, result, option_a);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 2) {
    HICE_CHECK(option_a == kNoTrans)
        << "Sparse transpose is not support for on cuda.";
    int64_t dim_m = tensor1.dim(0);
    int64_t dim_n = option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
    std::vector<int64_t> dims_result = {dim_m, dim_n};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    Tensor tensor2_new =
        option_b == kTrans ? contiguous(transpose_matrix(tensor2)) : tensor2;
    spmm(tensor1, tensor2_new, result);
  } else {
    HICE_CHECK(false) << "For sparse-dense matmul, Only mv and mm is "
                         "supported, but the inputs are "
                      << ndim_tensor1 << "D and " << ndim_tensor2 << "D";
  }
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl,
                     {kCUDA, kCOO},  // first operand
                     {kCUDA, kDense},   // second operand
                     {kCUDA, kDense}    // result
);

}  // namespace hice
#endif