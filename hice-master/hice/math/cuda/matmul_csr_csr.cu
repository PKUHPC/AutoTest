#if 0

#include "hice/basic/transpose.h"
#include "hice/device/cuda/context_cuda.h"
#include "hice/device/cuda/allocator_cuda.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/cuda/matmul_sparse_util.cuh"
#include "hice/math/matmul.h"

namespace hice {

namespace {

// csr = matmul(csr, csr)
void spgemm(const Tensor &tensor1, const Tensor &tensor2, Tensor &result) {
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
  cusparseHandle_t handle = cuda_ctx.cusparse_handle();
  HICE_CUSPARSE_CHECK(
      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  // sparse mat desc
  cusparseMatDescr_t descr = 0;
  HICE_CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  HICE_CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  HICE_CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  // infos about sparseA and sparseB
  int nnzA = tensor1.n_nonzero();
  int nnzB = tensor2.n_nonzero();
  // col indices row offsets of A and B
  const int *colindA_csr = tensor1.column_indices().data<int>();
  const int *rowindA_csr = tensor1.row_offsets().data<int>();
  const int *colindB_csr = tensor2.column_indices().data<int>();
  const int *rowindB_csr = tensor2.row_offsets().data<int>();
  // Allocate memory for row indices of C
  int nnzC;
  // Allocate memory for col indices of C
  int *rowindC_csr = result.mutable_row_offsets().mutable_data<int>();
  HICE_CUSPARSE_CHECK(cusparseXcsrgemmNnz(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    m, n, k,
    descr, nnzA, rowindA_csr, colindA_csr,
    descr, nnzB, rowindB_csr, colindB_csr,
    descr, rowindC_csr, &nnzC));
  // resize
  result.resize_with_nnz(nnzC);
  int *colindC_csr = result.mutable_column_indices().mutable_data<int>();
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      const scalar_t *valuesA = tensor1.values().data<scalar_t>();
      const scalar_t *valuesB = tensor2.values().data<scalar_t>();
      scalar_t *valuesC = result.mutable_values().mutable_data<scalar_t>();
      HICE_CUSPARSE_CHECK(cusparseScsrgemm(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, k,
        descr, nnzA, valuesA, rowindA_csr, colindA_csr,
        descr, nnzB, valuesB, rowindB_csr, colindB_csr,
        descr, valuesC, rowindC_csr, colindC_csr));
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      const scalar_t *valuesA = tensor1.values().data<scalar_t>();
      const scalar_t *valuesB = tensor2.values().data<scalar_t>();
      scalar_t *valuesC = result.mutable_values().mutable_data<scalar_t>();
      HICE_CUSPARSE_CHECK(cusparseDcsrgemm(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, k,
        descr, nnzA, valuesA, rowindA_csr, colindA_csr,
        descr, nnzB, valuesB, rowindB_csr, colindB_csr,
        descr, valuesC, rowindC_csr, colindC_csr));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  result.set_coalesced(true);
}

#if 0
/// NOTE: bufferSize get from cusparseScsrgemm2_bufferSizeExt is not correct.
// sparse = matmul(sparse, sparse)
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
  cusparseHandle_t handle = cuda_ctx.cusparse_handle();
  HICE_CUSPARSE_CHECK(
      cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  // sparse mat desc
  cusparseMatDescr_t descr = 0;
  HICE_CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  HICE_CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  HICE_CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  // create an opaque structure
  csrgemm2Info_t info;
  cusparseCreateCsrgemm2Info(&info);
  // infos about sparseA and sparseB
  int nnzA = get_n_nonzeros(tensor1);
  int nnzB = get_n_nonzeros(tensor2);
  Tensor indicesA = get_indices(tensor1).to(kInt32);
  Tensor indicesB = get_indices(tensor2).to(kInt32);
  const int *rowindA_coo = indicesA.data<int>();
  const int *rowindB_coo = indicesB.data<int>();
  const int *colindA = indicesA.data<int>() + nnzA;
  const int *colindB = indicesB.data<int>() + nnzB;
  int *rowindA_csr = NULL;
  int *rowindB_csr = NULL;
  cudaMalloc((void **)&rowindA_csr, sizeof(int) * (n_rows_tensor1 + 1));
  cudaMalloc((void **)&rowindB_csr, sizeof(int) * (n_rows_tensor2 + 1));
  HICE_CUSPARSE_CHECK(cusparseXcoo2csr(handle, rowindA_coo, nnzA,
                                       n_rows_tensor1, rowindA_csr,
                                       CUSPARSE_INDEX_BASE_ZERO));
  HICE_CUSPARSE_CHECK(cusparseXcoo2csr(handle, rowindB_coo, nnzB,
                                       n_rows_tensor2, rowindB_csr,
                                       CUSPARSE_INDEX_BASE_ZERO));
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      scalar_t alpha = 1.0;
      scalar_t *beta = NULL;  // if beta = NULL, C = alpha*A*B, D is not used
      const scalar_t *valuesA = get_values(tensor1).data<scalar_t>();
      const scalar_t *valuesB = get_values(tensor2).data<scalar_t>();
      // infos about sparse C
      int baseC, nnzC;
      size_t bufferSize = 0;
      void *buffer = NULL;
      int *rowindC_csr = NULL;
      int *colindC = NULL;
      scalar_t *valuesC = NULL;
      // nnzTotalDevHostPtr points to host memory
      int *nnzTotalDevHostPtr = &nnzC;
      // allocate buffer for csrgemm2Nnz and csrgemm2
      cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, descr, nnzA,
                                      rowindA_csr, colindA, descr, nnzB,
                                      rowindB_csr, colindB, beta, descr, nnzA,
                                      rowindA_csr, colindA, info, &bufferSize);
      {
        std::cout<<"nnzA="<<nnzA<<", nnzB="<<nnzB<<std::endl;
        std::cout<<"m="<<m<<", k="<<k<<", n="<<n<<std::endl;
        std::cout<<"bufferSize="<<bufferSize<<std::endl;

        print_cuda_array(rowindA_coo, nnzA, "rowindA_coo");
        print_cuda_array(rowindB_coo, nnzB, "rowindB_coo");
        print_cuda_array(rowindA_csr, m+1, "rowindA_csr");
        print_cuda_array(rowindB_csr, k+1, "rowindB_csr");
        print_cuda_array(colindA, nnzA, "colindA");
        print_cuda_array(colindB, nnzB, "colindB");

      }
      cudaMalloc(&buffer, bufferSize);
      // compute csrRowPtrC
      cudaMalloc((void **)&rowindC_csr, sizeof(int) * (m + 1));
      cusparseXcsrgemm2Nnz(handle, m, n, k, descr, nnzA, rowindA_csr, colindA,
                           descr, nnzB, rowindB_csr, colindB, descr, nnzA, rowindA_csr,
                           colindA, descr, rowindC_csr, nnzTotalDevHostPtr, info,
                           buffer);
      if (NULL != nnzTotalDevHostPtr) {
        nnzC = *nnzTotalDevHostPtr;
      } else {
        cudaMemcpy(&nnzC, rowindC_csr + m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, rowindC_csr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
      }
      // finish sparsity pattern and value of C
      cudaMalloc((void **)&colindC, sizeof(int) * nnzC);
      cudaMalloc((void **)&valuesC, sizeof(scalar_t) * nnzC);
      // Remark: set csrValC to null if only sparsity pattern is required.
      cusparseScsrgemm2(handle, m, n, k,
                        &alpha,
                        descr, nnzA, valuesA, rowindA_csr, colindA,   // A
                        descr, nnzB, valuesB, rowindB_csr, colindB,   // B
                        beta,
                        descr, nnzA, valuesA, rowindA_csr, colindA,   // D
                        descr, valuesC, rowindC_csr, colindC,         // C
                        info, buffer);
      print_cuda_array(valuesC, nnzC, "valuesC");
      print_cuda_array(rowindC_csr, m + 1, "rowindC_csr");
      print_cuda_array(colindC, nnzC, "colindC");
      // destroy the opaque structure
      cusparseDestroyCsrgemm2Info(info);
      cudaFree(buffer);
      cudaFree(rowindC_csr);
      cudaFree(colindC);
      cudaFree(valuesC);
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than "
                         "float, double, complex<float>, complex<double>";
  }
  // cudaFree(rowindA_csr);
  // cudaFree(rowindB_csr);
}
#endif

// sparse = matmul(sparse, sparse)
void matmul_spgemm_impl(const Tensor &tensor1_, const Tensor &tensor2_,
                      Tensor &result, MatmulOption option_a,
                      MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_cpu_sparse_sparse_sparse" << std::endl;
  Tensor tensor1 = tensor1_.is_coalesced() ? tensor1_ : tensor1_.to_coalesced();
  Tensor tensor2 = tensor2_.is_coalesced() ? tensor2_ : tensor2_.to_coalesced();
  // Tensor tensor1 = tensor1_;
  // Tensor tensor2 = tensor2_;
  const int64_t ndim_tensor1 = tensor1.ndim();
  const int64_t ndim_tensor2 = tensor2.ndim();
  HICE_CHECK(option_a == kNoTrans && option_b == kNoTrans)
      << "Sparse transpose is not support for on cuda.";
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
  spgemm(tensor1, tensor2, result);
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_spgemm_impl,
                     {kCUDA, kCSR},  // first operand
                     {kCUDA, kCSR},  // second operand
                     {kCUDA, kCSR}   // result
);

}  // namespace hice

#endif
