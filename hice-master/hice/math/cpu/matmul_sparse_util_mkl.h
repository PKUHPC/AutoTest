#pragma once
#ifdef HICE_USE_MKL
// #if 1

#include "hice/core/tensor.h"
// #include "hice/core/tensor_printer.h"
#include "hice/core/sparse_tensor.h"
#include "hice/device/cpu/allocator_cpu.h"
#include "hice/math/matmul.h"
#include "hice/device/cpu/common_mkl.h"


namespace hice {

namespace {

inline sparse_operation_t trans_option_from_(MatmulOption option) {
  if (option == kNoTrans) {
    return SPARSE_OPERATION_NON_TRANSPOSE;
  } else if (option == kTrans) {
    return SPARSE_OPERATION_TRANSPOSE;
  } else {
    return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  }
}

// load mkl coo from hice::sparse_tensor
inline void coo_from_(const Tensor &self, sparse_matrix_t *cooA) {
  Tensor& values = const_cast<Tensor &>(self).mutable_values();
  Tensor& indices = const_cast<Tensor &>(self).mutable_indices();
  const int n_nonzeros = self.n_nonzero();
  int *rowind = indices.mutable_data<int>();
  int *colind = indices.mutable_data<int>() + n_nonzeros;
  // create mkl_sparse from tensor
  sparse_matrix_t temp_coo;
  const int n_rows = self.dim(0);
  const int n_cols = self.dim(1);
  switch (self.scalar_type()) {
    case kFloat: {
      using scalar_t = float;
      scalar_t *val = values.mutable_data<scalar_t>();
      mkl_sparse_s_create_coo(&temp_coo, SPARSE_INDEX_BASE_ZERO, n_rows, n_cols,
                              n_nonzeros, rowind, colind, val);
      break;
    }
    case kDouble: {
      using scalar_t = double;
      scalar_t *val = values.mutable_data<scalar_t>();
      mkl_sparse_d_create_coo(&temp_coo, SPARSE_INDEX_BASE_ZERO, n_rows, n_cols,
                              n_nonzeros, rowind, colind, val);
      break;
    }
    default:
      HICE_LOG(ERROR) << "Not supported scalar type.";
  }
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  // the rowind and colind info in temp_coo(which comes from values and indices) may be
  // changed when doing sparse order by mkl, so do clone here
  mkl_sparse_copy(temp_coo, descr, cooA);
  mkl_sparse_destroy(temp_coo);
}

// load mkl csr from hice::sparse_tensor
inline void csr_from_(
    const Tensor &self, sparse_matrix_t *csr,
    sparse_operation_t trans = SPARSE_OPERATION_NON_TRANSPOSE) {
  sparse_matrix_t coo;
  coo_from_(self, &coo);
  mkl_sparse_convert_csr(coo, trans, csr);
  mkl_sparse_destroy(coo);
}

// load mkl csr from hice::csr_tensor
inline void mklcsr_from_csr(
    const Tensor &self, sparse_matrix_t *csr,
    sparse_operation_t trans = SPARSE_OPERATION_NON_TRANSPOSE) {
      Tensor& values = const_cast<Tensor &>(self).mutable_values();
      Tensor& column_indices = const_cast<Tensor &>(self).mutable_column_indices();
      Tensor& row_offsets = const_cast<Tensor &>(self).mutable_row_offsets();
      const int n_nonzeros = self.n_nonzero();
      int *row_offsets_ptr = row_offsets.mutable_data<int>();
      int *rows_start = row_offsets_ptr;
      int *rows_end = row_offsets_ptr + 1;
      int *colind = column_indices.mutable_data<int>();
      // create mkl_csr from csr tensor
      sparse_matrix_t temp_csr;
      const int n_rows = self.dim(0);
      const int n_cols = self.dim(1);
      switch (self.scalar_type()) {
        case kFloat: {
          using scalar_t = float;
          scalar_t *val = values.mutable_data<scalar_t>();
          mkl_sparse_s_create_csr(&temp_csr, SPARSE_INDEX_BASE_ZERO, n_rows, n_cols,
                                  rows_start, rows_end, colind, val);
          break;
        }
        case kDouble: {
          using scalar_t = double;
          scalar_t *val = values.mutable_data<scalar_t>();
          mkl_sparse_d_create_csr(&temp_csr, SPARSE_INDEX_BASE_ZERO, n_rows, n_cols,
                                  rows_start, rows_end, colind, val);
          break;
        }
        default:
          HICE_LOG(ERROR) << "Not supported scalar type.";
      }
      struct matrix_descr descr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      // the rowind and colind info in temp_csr(which comes from values and indices) may be
      // changed when doing sparse order by mkl, so do clone here
      mkl_sparse_copy(temp_csr, descr, csr);
      mkl_sparse_destroy(temp_csr);
}
// store mkl csr into hice::sparse_tensor
inline void store_csr_into_(const sparse_matrix_t source, Tensor &self) {
  int n_rows = -1;
  int n_cols = -1;
  int *rows_start = NULL;
  int *rows_end = NULL;
  int *col_indx = NULL;
  void *values_mkl = NULL;
  sparse_index_base_t indexing;
  int n_nonzero = 0;
  switch (self.scalar_type()) {
    case kFloat: {
      using scalar_t = float;
      mkl_sparse_s_export_csr(source, &indexing, &n_rows, &n_cols, &rows_start,
                              &rows_end, &col_indx, (scalar_t **)&values_mkl);
      // count nonzero elements
      n_nonzero = rows_end[n_rows - 1];
      // resize
      self.resize_with_nnz(n_nonzero);
      int32_t *indices_ptr = self.mutable_indices().mutable_data<int32_t>();
      scalar_t *values_ptr = self.mutable_values().mutable_data<scalar_t>();
      // get col_indices
      int64_t p = 0;
      for (int i = 0; i < n_rows; ++i) {
        for (int j = rows_start[i]; j < rows_end[i]; ++j) {
          indices_ptr[p++] = i;
        }
      }
      // get row_indices and values
      for (int i = 0; i < n_nonzero; ++i) {
        indices_ptr[p++] = col_indx[i];
        values_ptr[i] = ((scalar_t *)values_mkl)[i];
      }
      break;
    }
    case kDouble: {
      using scalar_t = double;
      mkl_sparse_d_export_csr(source, &indexing, &n_rows, &n_cols, &rows_start,
                              &rows_end, &col_indx, (scalar_t **)&values_mkl);
      // count nonzero elements
      n_nonzero = rows_end[n_rows - 1];
      // resize
      self.resize_with_nnz(n_nonzero);
      int32_t *indices_ptr = self.mutable_indices().mutable_data<int32_t>();
      scalar_t *values_ptr = self.mutable_values().mutable_data<scalar_t>();
      // get row_indices
      int64_t p = 0;
      for (int i = 0; i < n_rows; ++i) {
        for (int j = rows_start[i]; j < rows_end[i]; ++j) {
          indices_ptr[p++] = i;
        }
      }
      // get col_indices and values
      for (int i = 0; i < n_nonzero; ++i) {
        indices_ptr[p++] = col_indx[i];
        values_ptr[i] = ((scalar_t *)values_mkl)[i];
      }
      break;
    }
    default:
      HICE_LOG(ERROR) << "Not supported scalar type.";
  }
  HICE_CHECK(indexing == SPARSE_INDEX_BASE_ZERO);
}

inline void store_mklcsr_into_csr(const sparse_matrix_t source, Tensor &self) {
  int n_rows = -1;
  int n_cols = -1;
  int *rows_start = NULL;
  int *rows_end = NULL;
  int *col_indx = NULL;
  void *values_mkl = NULL;
  sparse_index_base_t indexing;
  int n_nonzero = 0;
  switch (self.scalar_type()) {
    case kFloat: {
      using scalar_t = float;
      mkl_sparse_s_export_csr(source, &indexing, &n_rows, &n_cols, &rows_start,
                              &rows_end, &col_indx, (scalar_t **)&values_mkl);
      // count nonzero elements
      n_nonzero = rows_end[n_rows - 1];
      // resize
      self.resize_with_nnz(n_nonzero);
      int32_t *column_indices_ptr = self.mutable_column_indices().mutable_data<int32_t>();
      int32_t *row_offsets_ptr = self.mutable_row_offsets().mutable_data<int32_t>();
      scalar_t *values_ptr = self.mutable_values().mutable_data<scalar_t>();
      // get col_indices
      int64_t p = 0;
      for (int i = 0; i < n_rows; ++i) {
          row_offsets_ptr[i] = rows_start[i];
      }
      row_offsets_ptr[n_rows] = rows_end[n_rows - 1];
      // get column_indices and values
      for (int i = 0; i < n_nonzero; ++i) {
        column_indices_ptr[i] = col_indx[i];
        values_ptr[i] = ((scalar_t *)values_mkl)[i];
      }
      break;
    }
    case kDouble: {
      using scalar_t = double;
      mkl_sparse_d_export_csr(source, &indexing, &n_rows, &n_cols, &rows_start,
                              &rows_end, &col_indx, (scalar_t **)&values_mkl);
      // count nonzero elements
      n_nonzero = rows_end[n_rows - 1];
      // resize
      self.resize_with_nnz(n_nonzero);
      int32_t *column_indices_ptr = self.mutable_column_indices().mutable_data<int32_t>();
      int32_t *row_offsets_ptr = self.mutable_row_offsets().mutable_data<int32_t>();
      scalar_t *values_ptr = self.mutable_values().mutable_data<scalar_t>();
      // get col_indices
      int64_t p = 0;
      for (int i = 0; i < n_rows; ++i) {
          row_offsets_ptr[i] = rows_start[i];
      }
      row_offsets_ptr[n_rows] = rows_end[n_rows - 1];
      // get column_indices and values
      for (int i = 0; i < n_nonzero; ++i) {
        column_indices_ptr[i] = col_indx[i];
        values_ptr[i] = ((scalar_t *)values_mkl)[i];
      }
    }
    default:
      HICE_LOG(ERROR) << "Not supported scalar type.";
  }
  HICE_CHECK(indexing == SPARSE_INDEX_BASE_ZERO);
}
}
}


#endif
