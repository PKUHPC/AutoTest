#pragma once

#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "mkl.h"

#include "timer.h"

using namespace hice;

void gemv_mklblas(const Tensor &mat, const Tensor &vec, Tensor &result) {
  const int n_rows = mat.dim(0);
  const int n_cols = mat.dim(1);
  const int length = vec.dim(0);
  const int lda = n_cols;
  const int incx = 1;
  const int incy = 1;
  ScalarType sc_type = mat.scalar_type();
  switch(sc_type) {
    case kFloat: {
      using scalar_t = float;
      const scalar_t *a = mat.data<scalar_t>();
      const scalar_t *x = vec.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cblas_sgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, alpha, a, lda, x,
                  incx, beta, y, incy);
      break;
    }
    case kDouble: {
      using scalar_t = double;
      const scalar_t *a = mat.data<scalar_t>();
      const scalar_t *x = vec.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cblas_dgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, alpha, a, lda, x,
                  incx, beta, y, incy);
      break;
    }
  }
}

void gemm_mklblas(const Tensor &mat1, const Tensor &mat2, Tensor &result) {
  const int n_rows_mat1 = mat1.dim(0);
  const int n_cols_mat1 = mat1.dim(1);
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  const int m = n_rows_mat1;
  const int k = n_rows_mat2;
  const int n = n_cols_mat2;
  const int lda = n_cols_mat1;
  const int ldb = n_cols_mat2;
  const int ldc = result.dim(1);
  ScalarType sc_type = mat1.scalar_type();
  switch(sc_type) {
    case kFloat: {
      using scalar_t = float;
      const scalar_t *a = mat1.data<scalar_t>();
      const scalar_t *b = mat2.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda,
                  b, ldb, beta, c, ldc);
      break;
    }
    case kDouble: {
      using scalar_t = double;
      const scalar_t *a = mat1.data<scalar_t>();
      const scalar_t *b = mat2.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda,
                  b, ldb, beta, c, ldc);
      break;
    }
  }
}

void gemv_cublas(const Tensor &mat, const Tensor &vec, Tensor &result) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  const int n_rows = mat.dim(0);
  const int n_cols = mat.dim(1);
  const int m = n_cols;
  const int lda = n_cols;
  const int incx = 1;
  const int incy = 1;
  ScalarType sc_type = mat.scalar_type();
  switch(sc_type) {
    case kFloat: {
      using scalar_t = float;
      const scalar_t *a = mat.data<scalar_t>();
      const scalar_t *x = vec.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cublasSgemv(handle, CUBLAS_OP_T, n_cols, n_rows, &alpha, a,
                                    lda, x, incx, &beta, y, incy);
      break;
    }
    case kDouble: {
      using scalar_t = double;
      const scalar_t *a = mat.data<scalar_t>();
      const scalar_t *x = vec.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cublasDgemv(handle, CUBLAS_OP_T, n_cols, n_rows, &alpha, a,
                                    lda, x, incx, &beta, y, incy);
      break;
    }
  }
}



void gemm_cublas(const Tensor &mat1, const Tensor &mat2, Tensor &result) {
  // static cublasHandle_t handle;
  // static bool init = false;
  // if (!init) {
  //   cublasCreate(&handle);
  //   init = true;
  // }
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  const int n_rows_mat1 = mat1.dim(0);
  const int n_cols_mat1 = mat1.dim(1);
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  const int n_cols_result = result.dim(-1);
  const int m = n_cols_mat2;
  const int k = n_rows_mat2;
  const int n = n_rows_mat1;
  const int lda = n_cols_mat2;
  const int ldb = n_cols_mat1;
  const int ldc = n_cols_result;
  ScalarType sc_type = mat1.scalar_type();
  switch(sc_type) {
    case kFloat: {
      using scalar_t = float;
      const scalar_t *a = mat2.data<scalar_t>();
      const scalar_t *b = mat1.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb,
                  &beta, c, ldc);
      break;
    }
    case kDouble: {
      using scalar_t = double;
      const scalar_t *a = mat2.data<scalar_t>();
      const scalar_t *b = mat1.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb,
                  &beta, c, ldc);
      break;
    }
  }
}



Tensor gemv_mklblas(const Tensor &mat, const Tensor &vec) {
  const int n_rows = mat.dim(0);
  const int n_cols = mat.dim(1);
  const int length = vec.dim(0);
  Tensor result({n_rows}, vec.options());
  const int lda = n_cols;
  const int incx = 1;
  const int incy = 1;
  const double *a = mat.data<double>();
  const double *x = vec.data<double>();
  double *y = result.mutable_data<double>();
  double alpha = 1, beta = 0;
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n_rows, n_cols, alpha, a, lda, x,
              incx, beta, y, incy);
  return result;
}

Tensor gemm_mklblas(const Tensor &mat1, const Tensor &mat2) {
  const int n_rows_mat1 = mat1.dim(0);
  const int n_cols_mat1 = mat1.dim(1);
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  Tensor result({n_rows_mat1, n_cols_mat2}, mat1.options());
  const int m = n_rows_mat1;
  const int k = n_rows_mat2;
  const int n = n_cols_mat2;
  const int lda = n_cols_mat1;
  const int ldb = n_cols_mat2;
  const int ldc = result.dim(1);
  const double *a = mat1.data<double>();
  const double *b = mat2.data<double>();
  double *c = result.mutable_data<double>();
  double alpha = 1, beta = 0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda,
              b, ldb, beta, c, ldc);
  return result;
}

Tensor gemv_cublas(const Tensor &mat, const Tensor &vec) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  const int n_rows = mat.dim(0);
  const int n_cols = mat.dim(1);
  const int m = n_cols;
  const int lda = n_cols;
  const int incx = 1;
  const int incy = 1;
  Tensor result({n_rows}, vec.options());
  const double *a = mat.data<double>();
  const double *x = vec.data<double>();
  double *y = result.mutable_data<double>();
  double alpha = 1, beta = 0;
  cublasDgemv(handle, CUBLAS_OP_T, n_cols, n_rows, &alpha, a,
                                lda, x, incx, &beta, y, incy);
  return result;
}

Tensor gemm_cublas(const Tensor &mat1, const Tensor &mat2) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  const int n_rows_mat1 = mat1.dim(0);
  const int n_cols_mat1 = mat1.dim(1);
  const int n_rows_mat2 = mat2.dim(0);
  const int n_cols_mat2 = mat2.dim(1);
  Tensor result({n_rows_mat1, n_cols_mat2}, mat1.options());
  const int n_cols_result = result.dim(-1);
  const int m = n_cols_mat2;
  const int k = n_rows_mat2;
  const int n = n_rows_mat1;
  const int lda = n_cols_mat2;
  const int ldb = n_cols_mat1;
  const int ldc = n_cols_result;
  const double *a = mat2.data<double>();
  const double *b = mat1.data<double>();
  double *c = result.mutable_data<double>();
  double alpha = 1, beta = 0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, lda, b, ldb,
              &beta, c, ldc);
  return result;
}