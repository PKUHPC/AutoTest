#include "hice/core/sparse_tensor.h"
#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/core/dispatch.h"
#include "hice/core/shape_util.h"

namespace hice {

const SparseTensorImplCSR& get_impl_csr(const Tensor& self) {
  HICE_CHECK(self.is_csr()) << "get_impl_csr: not a csr sparse tensor";
  // use static_cast(instead of dynamic_cast) to decrease the runtime cost
  return static_cast<const SparseTensorImplCSR&>(self.impl());
}

SparseTensorImplCSR& get_mutable_impl_csr(Tensor& self) {
  HICE_CHECK(self.is_csr()) << "get_mutable_impl_csr: not a csr sparse tensor";
  return static_cast<SparseTensorImplCSR&>(self.mutable_impl());
}
// create an empty sparse tensor in csr format
Tensor new_csr(const TensorOptions& options) {
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCSR);
  return make_tensor<SparseTensorImplCSR>(options.layout(kCSR));
}

// create an empty sparse tensor with dims in csr format
Tensor new_csr(ConstIntArrayRef dims, const TensorOptions& options) {
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCSR);
  int64_t ndim = dims.size();
  HICE_CHECK(ndim <= 2)
      << "Can not creat a new csr tensor with over 2 dimensions";
  Tensor self = new_csr(options);
  self.resize(dims);
  return self;
}

Tensor new_csr(ConstIntArrayRef dims, const Tensor& column_indices,
               const Tensor& row_offsets, const Tensor& values,
               const TensorOptions& options, bool copy_) {
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCSR);
  int64_t ndim = dims.size();
  int64_t n_rows = dims[0];
  HICE_CHECK(!(ndim == 1 && dims[0] == 0))
      << "Can not creat a csr scalar tensor";
  HICE_CHECK(ndim <= 2)
      << "Can not creat a new csr tensor with over 2 dimensions";
  HICE_CHECK(values.is_dense());
  HICE_CHECK(column_indices.is_dense());
  HICE_CHECK(row_offsets.is_dense());

  HICE_CHECK(values.ndim() == 1);
  HICE_CHECK(column_indices.ndim() == 1);
  HICE_CHECK(row_offsets.ndim() == 1);

  HICE_CHECK(row_offsets.size() == (n_rows + 1));
  HICE_CHECK(column_indices.size() == values.size());

  Tensor self;
  if (copy_) {
    Tensor new_column_indices(column_indices.dims(),
                              column_indices.options());
    Tensor new_row_offsets(row_offsets.dims(), row_offsets.options());
    Tensor new_values(values.dims(), values.options());
    hice::copy(column_indices, new_column_indices);
    hice::copy(row_offsets, new_row_offsets);
    hice::copy(values, new_values);
    self = make_tensor<SparseTensorImplCSR>(
        options.layout(kCSR), new_column_indices, new_row_offsets, new_values);
  } else {
    self = make_tensor<SparseTensorImplCSR>(
        options.layout(kCSR), column_indices, row_offsets, values);
  }
  SparseTensorImplCSR& csr_impl = get_mutable_impl_csr(self);
  // set shape
  auto shape = ShapeUtil::make_shape(dims);
  shape.mutable_layout().set_type(kCSR);
  csr_impl.set_shape(shape);
  return self;
}
Tensor wrap_csr(ConstIntArrayRef dims, int32_t* column_indices_ptr,
                int32_t* row_offsets_ptr, void* values_ptr,
                const int64_t n_nonzero, const TensorOptions& options,
                bool copy_) {
  int64_t n_rows = dims[0];
  std::vector<int64_t> dims_values = {n_nonzero};
  std::vector<int64_t> dims_column_indices = {n_nonzero};
  std::vector<int64_t> dims_row_offsets = {n_rows + 1};
  Tensor column_indices = wrap(dims_column_indices, column_indices_ptr,
                               options.dtype(kInt32).layout(kDense), copy_);
  Tensor row_offsets = wrap(dims_row_offsets, row_offsets_ptr,
                            options.dtype(kInt32).layout(kDense), copy_);
  Tensor values = wrap(dims_values, values_ptr, options.layout(kDense), copy_);
  return new_csr(dims, column_indices, row_offsets, values, options,
                 false);  // no need to copy any more
}

Tensor dense_to_csr(const Tensor& self_) {
  HICE_CHECK(self_.is_dense());
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  int64_t n_rows = dims[0];
  HICE_CHECK(!(ndim == 1 && dims[0] == 0))
      << "Can not convert a scalar tensor to csr.";
  HICE_CHECK(ndim <= 2)
      << "Can not convert a dense tensor with over 2 dimensions to csr format";
  if (ndim == 0) {
    return new_csr(self_.options().layout(kCSR));
  }
  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  Tensor csr = new_csr(dims, self.options().layout(kCSR));
  int64_t nnz = 0;
  // count non-zeros
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "dense_to_csr_count_nnz",
                          [&]() {
                            const scalar_t* dense_ptr = self.data<scalar_t>();
                            int64_t size = self.size();
                            for (int i = 0; i < size; ++i) {
                              if (dense_ptr[i] != 0) {
                                nnz += 1;
                              }
                            }
                          });
  csr.resize_with_nnz(nnz);
#if 1
  // set values
  HICE_DISPATCH_ALL_TYPES(
      self.scalar_type(), "dense_to_csr_set_values", [&]() {
        const scalar_t* dense_ptr = self.data<scalar_t>();
        int* row_offsets_ptr = csr.mutable_row_offsets().mutable_data<int>();
        int* column_indices_ptr = csr.mutable_column_indices().mutable_data<int>();
        scalar_t* values_ptr = csr.mutable_values().mutable_data<scalar_t>();
        int64_t size = self.size();
        // store number of nnz in every single row
        std::vector<int64_t> nonezero_per_row(n_rows, 0);
        std::vector<int64_t> dense_idx(ndim, 0);
        std::vector<int64_t> strides = self.strides();
        int64_t sparse_idx = 0;
        for (int i = 0; i < size; ++i) {
          int64_t offset = 0;
          for (int j = ndim - 1; j >= 0; --j) {
            offset += strides[j] * dense_idx[j];
          }
          // set non-zero
          if (dense_ptr[offset] != 0) {
            int row_id =  dense_idx[0];
            nonezero_per_row[row_id] ++;
            values_ptr[sparse_idx] = dense_ptr[offset];
            if(ndim > 1){
              column_indices_ptr[sparse_idx] = dense_idx[1];
            }else {
              column_indices_ptr[sparse_idx] = 0;
            }
            sparse_idx += 1;
          }
          // update dense_idx
          for (int j = ndim - 1; j >= 0; --j) {
            dense_idx[j] += 1;
            if (dense_idx[j] < dims[j]) {
              break;
            } else {
              dense_idx[j] = 0;
            }
          }
        }
        row_offsets_ptr[0] = 0;
        for(int i = 0; i < n_rows; i++) {
          row_offsets_ptr[i + 1] = row_offsets_ptr[i] + nonezero_per_row[i];
        }
      });
  csr.set_coalesced(true);
  if (self_.device().is_cuda()) {
    return csr.to(kCUDA);
  } else {
    return csr;
  }
#endif
}

Tensor csr_to_dense(const Tensor& self_) {
  HICE_CHECK(self_.is_csr());
  int64_t nnz = self_.n_nonzero();
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  int64_t n_rows = dims[0];
  int64_t n_cols = 0;
  if(ndim > 1) {
    n_cols = dims[1];
  } else {
    n_cols = 0;
  }
  if (ndim == 0) {
    return empty({}, self_.options().layout(kDense));
  } else if (nnz == 0) {
    return full(dims, 0, self_.options().layout(kDense));
  }
  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  Tensor dense = full(dims, 0, self.options().layout(kDense));
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "csr_to_dense", [&]() {
    const int* row_offsets_ptr = self.row_offsets().data<int>();
    const int* column_indices_ptr = self.column_indices().data<int>();
    const scalar_t* values_ptr = self.values().data<scalar_t>();
    scalar_t* dense_ptr = dense.mutable_data<scalar_t>();
    for (int i = 0; i < n_rows; ++i) {
      int64_t row_id = i;
      int64_t row_start = row_offsets_ptr[i];
      int64_t row_end = row_offsets_ptr[i + 1];
      for (int j = row_start; j < row_end; j++) {
        int64_t col_ind = column_indices_ptr[j];
        dense_ptr[row_id * n_cols + col_ind] = values_ptr[j];
      }
    }
  });
  if (self_.device().is_cuda()) {
    return dense.to(kCUDA);
  } else {
    return dense;
  }
}

Tensor csr_to_sparse(const Tensor& self_) {
  HICE_CHECK(self_.is_csr());
  int64_t nnz = self_.n_nonzero();
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  int64_t n_rows = dims[0];
  int64_t n_cols = 0;
  if(ndim > 1) {
    n_cols = dims[1];
  } else {
    n_cols = 0;
  }
  if (ndim == 0) {
    return new_sparse(self_.options().layout(kCOO));
  }
  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  self = self.to_coalesced(); 
  Tensor coo = new_sparse(dims, self.options().layout(kCOO));
  coo.resize_with_nnz(nnz);
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "csr_to_coo", [&]() {
    const int* row_offsets_ptr = self.row_offsets().data<int>();
    const int* column_indices_ptr = self.column_indices().data<int>();
    const scalar_t* values_ptr = self.values().data<scalar_t>();
    Tensor coo_values = coo.mutable_values();
    Tensor coo_indices = coo.mutable_indices();
    int *coo_rowind = coo_indices.mutable_data<int>();
    int *coo_colind = coo_indices.mutable_data<int>() + nnz; 
    scalar_t* coo_values_ptr = coo_values.mutable_data<scalar_t>();
    for (int i = 0; i < n_rows; ++i) {
      int64_t row_id = i;
      int64_t row_start = row_offsets_ptr[i];
      int64_t row_end = row_offsets_ptr[i + 1];
      for (int j = row_start; j < row_end; j++) {
        coo_rowind[j] = row_id;
      }
    }
    for (size_t i = 0; i < nnz; i++) {
      coo_values_ptr[i] = values_ptr[i];
      coo_colind[i] = column_indices_ptr[i];
    }
  });
  coo.set_coalesced(true);
  if (self_.device().is_cuda()) {
    return coo.to(kCUDA);
  } else {
    return coo;
  }
}

}  // namespace hice
