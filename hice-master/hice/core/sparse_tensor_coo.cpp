#include "hice/core/sparse_tensor.h"
#include "hice/basic/copy.h"
#include "hice/core/dispatch.h"
#include "hice/core/shape_util.h"
#include "hice/math/binary_expr.h"
#include "hice/math/reduce.h"
#include "hice/basic/reshape.h"
#include "hice/basic/slice.h"

namespace hice {

const SparseTensorImplCOO& get_impl_coo(const Tensor& self) {
  HICE_CHECK(self.is_coo()) << "get_impl_coo: not a coo sparse tensor";
  return static_cast<const SparseTensorImplCOO&>(self.impl());
}

SparseTensorImplCOO& get_mutable_impl_coo(Tensor& self) {
  HICE_CHECK(self.is_coo()) << "get_mutable_impl_coo: not a coo sparse tensor";
  return static_cast<SparseTensorImplCOO&>(self.mutable_impl());
}

// create an empty sparse tensor in coo format
Tensor new_sparse(const TensorOptions& options) {
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCOO);
  return make_tensor<SparseTensorImplCOO>(options.layout(kCOO));
}

// create an empty sparse tensor with dims in coo format
Tensor new_sparse(ConstIntArrayRef dims, const TensorOptions& options) {
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCOO);
  Tensor self = new_sparse(options);
  self.resize(dims);
  return self;
}

// create an sparse tensor with dims, indices and values in coo format
Tensor new_sparse(ConstIntArrayRef dims, const Tensor& indices,
                  const Tensor& values, const TensorOptions& options,
                  bool copy_) {
  // arg checking
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCOO);
  // the following checks are redundant because they are also checked in
  // SparseTensorImplCOO::set_indices_and_values_unsafe but we need to ensure
  // them in order to infer the shape.
  HICE_CHECK(indices.ndim() == 2);
  HICE_CHECK(indices.is_dense());
  int64_t ndim = dims.size();
  HICE_CHECK(indices.dim(0) == ndim);

  // Check to make sure all indices are within the boundaries of `size`
  if (indices.size() > 0) {
    Tensor min_indices = reduce_min(indices, {1}, false);
    Tensor max_indices = reduce_max(indices, {1}, false);
    Tensor cpu_min_indices, cpu_max_indices;
    if (indices.device().is_cuda()) {
      cpu_min_indices = min_indices.to(kCPU);
      cpu_max_indices = max_indices.to(kCPU);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    const int32_t* cpu_min_indices_data = cpu_min_indices.data<int32_t>();
    const int32_t* cpu_max_indices_data = cpu_max_indices.data<int32_t>();
    for (int64_t d = 0; d < ndim; d++) {
      int64_t min_index_in_dim = cpu_min_indices_data[d];
      HICE_CHECK(min_index_in_dim >= 0)
          << "found negative index " << min_index_in_dim << " for dim " << d;
      int64_t max_index_in_dim = cpu_max_indices_data[d];
      int64_t dim_size = dims[static_cast<size_t>(d)];
      HICE_CHECK(max_index_in_dim < dim_size)
          << "size is inconsistent with indices: for dim " << d << ", size is "
          << dim_size << " but found index " << max_index_in_dim;
    }
  }
  return new_sparse_unsafe(dims, indices, values, options.layout(kCOO), copy_);
}

// create an sparse tensor with indices and values in coo format
Tensor new_sparse(const Tensor& indices, const Tensor& values,
                  const TensorOptions& options, bool copy_) {
  HICE_CHECK(values.ndim() == 1);
  HICE_CHECK(indices.ndim() == 2);
  HICE_CHECK(values.is_dense());
  HICE_CHECK(indices.is_dense());
  HICE_CHECK(!options.has_layout() || options.layout_type() == kCOO);
  // If dims are not given, it is inferred as max index of each dim.
  int64_t sparse_ndim = indices.dim(0);
  std::vector<int64_t> inferred_dims(sparse_ndim);
  if (indices.size() > 0) {
    // If the indices has elements in it, we infer the minimum sparse dimension
    // sizes as the max value of each dim in indices.
    Tensor min_indices = reduce_min(indices, {1}, false);
    Tensor inferred_indices_dims = reduce_max(indices, {1}, false);
    add(inferred_indices_dims, 1,
        inferred_indices_dims);  // len = max_index + 1
    Tensor cpu_min_indices = min_indices.to(kCPU);
    Tensor cpu_inferred_indices_dims = inferred_indices_dims.to(kCPU);
    const int32_t* cpu_min_indices_data = cpu_min_indices.data<int32_t>();
    const int64_t* cpu_inferred_indices_dims_data =
        cpu_inferred_indices_dims.data<int64_t>();
    for (int64_t d = 0; d < sparse_ndim; d++) {
      int64_t min_index_in_dim = cpu_min_indices_data[d];
      HICE_CHECK(min_index_in_dim >= 0)
          << "found negative index " << min_index_in_dim << " for dim " << d;
      inferred_dims[static_cast<size_t>(d)] = cpu_inferred_indices_dims_data[d];
    }
  } else {
    // If the indices doesn't have elements in it, there is not enough
    // information to know what the minimum sparse dimension sizes should be,
    // and in this case we set them to 0
    for (int64_t d = 0; d < sparse_ndim; d++) {
      inferred_dims[static_cast<size_t>(d)] = 0;
    }
  }
  return new_sparse_unsafe(inferred_dims, indices, values, options.layout(kCOO),
                           copy_);
}

Tensor new_sparse_unsafe(ConstIntArrayRef dims, const Tensor& indices,
                         const Tensor& values, const TensorOptions& options,
                         bool copy_) {
  Tensor self = new_sparse(options);
  self.resize(dims);
  SparseTensorImplCOO& spimpl = get_mutable_impl_coo(self);
  if (copy_) {
    Tensor new_indices(indices.dims(), self.indices().options());
    Tensor new_values(values.dims(), self.values().options());
    hice::copy(indices, new_indices);
    hice::copy(values, new_values);
    spimpl.set_indices_and_values_unsafe(new_indices, new_values);
  } else {
    spimpl.set_indices_and_values_unsafe(indices, values);
  }

  return self;
}

Tensor wrap_sparse(ConstIntArrayRef dims, int32_t* indices_ptr,
                   void* values_ptr, const int64_t n_nonzero,
                   const TensorOptions& options, bool copy_) {
  std::vector<int64_t> dims_values = {n_nonzero};
  std::vector<int64_t> dims_indices = {(int64_t)dims.size(), n_nonzero};
  Tensor indices = wrap(dims_indices, indices_ptr,
                        options.dtype(kInt32).layout(kDense), copy_);
  Tensor values = wrap(dims_values, values_ptr, options.layout(kDense), copy_);
  return new_sparse(dims, indices, values, options,
                    false);  // no need to copy any more
}

Tensor wrap_sparse_unsafe(ConstIntArrayRef dims, int32_t* indices_ptr,
                          void* values_ptr, const int64_t n_nonzero,
                          const TensorOptions& options, bool copy_) {
  std::vector<int64_t> dims_values = {n_nonzero};
  std::vector<int64_t> dims_indices = {(int64_t)dims.size(), n_nonzero};
  Tensor indices = wrap(dims_indices, indices_ptr,
                        options.dtype(kInt32).layout(kDense), copy_);
  Tensor values = wrap(dims_values, values_ptr, options.layout(kDense), copy_);
  return new_sparse_unsafe(dims, indices, values, options,
                           false);  // no need to copy any more
}

Tensor flatten_indices(const Tensor& indices, ConstIntArrayRef dims,
                       bool force_clone) {
  HICE_CHECK(indices.dim(0) == dims.size());
  int64_t sparse_ndim = indices.dim(0);
  if (sparse_ndim == 1) {
    if (force_clone) {
      return squeeze(indices, 0).clone();
    } else {
      return squeeze(indices, 0);
    }
  } else {
    std::vector<int64_t> indices_mult_cpu_vec;
    indices_mult_cpu_vec.reserve(sparse_ndim);
    int64_t mult = 1;
    for (int64_t i = sparse_ndim - 1; i >= 0; i--) {
      indices_mult_cpu_vec[i] = mult;
      mult *= dims[i];
    }
    Tensor indices_mult_cpu =
        wrap({sparse_ndim, 1}, indices_mult_cpu_vec.data(),
             indices.options().device(kCPU));
    Tensor indices_mult = indices_mult_cpu.to(indices.device());
    return reduce_sum(mul(indices, indices_mult), {0}, /* keep_dim */ false);
  }
}

Tensor flatten_indices_by_dims(const Tensor& indices,
                               const ConstIntArrayRef& dims,
                               const ConstIntArrayRef& dims_to_flatten) {
  Tensor new_indices = full({indices.dim(1)}, 0, indices.options());
  for (auto d : dims_to_flatten) {
    mul(new_indices, dims[d], new_indices);
    add(new_indices, slice(indices, 0, d, d + 1), new_indices);
  }
  return new_indices;
}

Tensor dense_to_sparse(const Tensor& self_) {
  HICE_CHECK(self_.is_dense());
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  HICE_CHECK(!(ndim == 1 && dims[0] == 0))
      << "Can not convert a scalar tensor to sparse.";
  if (ndim == 0) {
    return new_sparse(self_.options().layout(kCOO));
  }

  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  Tensor sparse = new_sparse(dims, self.options().layout(kCOO));
  int64_t nnz = 0;
  // count non-zeros
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "dense_to_sparse_count_nnz",
                          [&]() {
                            const scalar_t* dense_ptr = self.data<scalar_t>();
                            int64_t size = self.size();
                            for (int i = 0; i < size; ++i) {
                              if (dense_ptr[i] != 0) {
                                nnz += 1;
                              }
                            }
                          });
  sparse.resize_with_nnz(nnz);
  // set values
  HICE_DISPATCH_ALL_TYPES(
      self.scalar_type(), "dense_to_sparse_set_values", [&]() {
        const scalar_t* dense_ptr = self.data<scalar_t>();
        int* indices_ptr = sparse.mutable_indices().mutable_data<int>();
        scalar_t* values_ptr = sparse.mutable_values().mutable_data<scalar_t>();
        int64_t size = self.size();
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
            values_ptr[sparse_idx] = dense_ptr[offset];
            for (int j = ndim - 1; j >= 0; --j) {
              indices_ptr[j * nnz + sparse_idx] = dense_idx[j];
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
      });
  sparse.set_coalesced(true);
  if (self_.device().is_cuda()) {
    return sparse.to(kCUDA);
  } else {
    return sparse;
  }
}

Tensor sparse_to_dense(const Tensor& self_) {
  HICE_CHECK(self_.is_coo());
  int64_t nnz = self_.n_nonzero();
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  if (ndim == 0) {
    return empty({}, self_.options().layout(kDense));
  } else if (nnz == 0) {
    return full(dims, 0, self_.options().layout(kDense));
  }

  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  Tensor dense = full(dims, 0, self.options().layout(kDense));
  // get strides
  std::vector<int64_t> strides(ndim, 0);
  int64_t strd_ = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = strd_;
    strd_ *= dims[i];
  }
  // set non-zeros
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "sparse_to_dense", [&]() {
    const int* indices_ptr = self.indices().data<int>();
    const scalar_t* values_ptr = self.values().data<scalar_t>();
    scalar_t* dense_ptr = dense.mutable_data<scalar_t>();
    for (int i = 0; i < nnz; ++i) {
      int64_t offset = 0;
      for (int j = ndim - 1; j >= 0; --j) {
        offset += indices_ptr[j * nnz + i] * strides[j];
      }
      if (dense_ptr[offset] == 0) {
        dense_ptr[offset] = values_ptr[i];
      } else {
        dense_ptr[offset] += values_ptr[i];
      }
    }
  });
  if (self_.device().is_cuda()) {
    return dense.to(kCUDA);
  } else {
    return dense;
  }
}

Tensor sparse_to_csr(const Tensor& self_) {
  HICE_CHECK(self_.is_coo());
  int64_t nnz = self_.n_nonzero();
  int64_t ndim = self_.ndim();
  ConstIntArrayRef dims = self_.dims();
  int64_t n_rows = dims[0];
  HICE_CHECK(!(ndim == 1 && dims[0] == 0))
      << "Can not convert a scalar tensor to csr.";
  HICE_CHECK(ndim <= 2)
      << "Can not convert a coo tensor with over 2 dimensions to csr format";
  if (ndim == 0) {
    return new_csr(self_.options().layout(kCSR));
  }
  Tensor self = self_.device().is_cuda() ? self_.to(kCPU) : self_;
  self = self.to_coalesced(); 
  Tensor& coo_values = const_cast<Tensor &>(self).mutable_values();
  Tensor& coo_indices = const_cast<Tensor &>(self).mutable_indices();
  int *coo_rowind = coo_indices.mutable_data<int>();
  int *coo_colind = coo_indices.mutable_data<int>() + nnz;
  std::vector<int> coo_rowind_vec(coo_rowind, coo_rowind + nnz);
  Tensor csr = new_csr(dims, self.options().layout(kCSR));
  csr.resize_with_nnz(nnz);
  int* row_offsets_ptr = csr.mutable_row_offsets().mutable_data<int>();
  row_offsets_ptr[0] = 0;  
  for (int i = 0; i < n_rows; i++) {
    // int nnz_per_row = std::count_if(std::begin(coo_rowind_vec), std::end(coo_rowind_vec), [=](int a) {
    //   return a == i; 
    // });
    int nnz_per_row = 0;
    for(int j = 0; j < coo_rowind_vec.size(); j++) {
      if(coo_rowind_vec[j] == i)
        nnz_per_row ++; 
    }
    row_offsets_ptr[i + 1] = nnz_per_row + row_offsets_ptr[i];
  }
  HICE_DISPATCH_ALL_TYPES(self.scalar_type(), "sparse_to_csr", [&]() {
    int* column_indices_ptr = csr.mutable_column_indices().mutable_data<int>();
    scalar_t* values_ptr = csr.mutable_values().mutable_data<scalar_t>();
    scalar_t* coo_values_ptr = coo_values.mutable_data<scalar_t>();
    for (size_t i = 0; i < nnz; i++) {
      values_ptr[i] = coo_values_ptr[i];
      column_indices_ptr[i] = coo_colind[i];
    }
  });
  csr.set_coalesced(true);
  if (self_.device().is_cuda()) {
    return csr.to(kCUDA);
  } else {
    return csr;
  }
}

}  // namespace hice
