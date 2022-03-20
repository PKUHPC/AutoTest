#pragma once

/// NOTE: HICE support sparse only in COO format for now. CSR is for developers.
#include "hice/core/sparse_tensor_impl_coo.h"
#include "hice/core/sparse_tensor_impl_csr.h"
#include "hice/core/tensor.h"

/// Sparse helper methods for users
namespace hice {

HICE_API const SparseTensorImplCOO& get_impl_coo(const Tensor& self);

HICE_API SparseTensorImplCOO& get_mutable_impl_coo(Tensor& self);

/// create an empty sparse tensor in coo format
HICE_API Tensor new_sparse(const TensorOptions& options);

/// create an empty sparse tensor with dims in coo format
HICE_API Tensor new_sparse(ConstIntArrayRef dims, const TensorOptions& options);

/// create an sparse tensor with dims, indices and values in coo format.
/// There will be data copy if copy_ is true.
///
/// It will set 'coalesced_' to false by default, see NOTE [ To Coalesced ]
HICE_API Tensor new_sparse(ConstIntArrayRef dims, const Tensor& indices,
                           const Tensor& values, const TensorOptions& options,
                           bool copy_ = false);

/// create an sparse tensor with indices and values in coo format.
/// There will be data copy if copy_ is true.
///
/// It will set 'coalesced_' to false by default, see NOTE [ To Coalesced ]
HICE_API Tensor new_sparse(const Tensor& indices, const Tensor& values,
                           const TensorOptions& options, bool copy_ = false);

/// NOTE: new_sparse_unsafe() differs from new_sparse()
/// in that we don't check whether any indices are out of boundaries of `size`,
/// thus avoiding a copy from CUDA to CPU. However, this function should ONLY be
/// used where we know that the indices are guaranteed to be within bounds.
/// There will be data copy if copy_ is true.
///
/// It will set 'coalesced_' to false by default, see NOTE [ To Coalesced ]
HICE_API Tensor new_sparse_unsafe(ConstIntArrayRef dims, const Tensor& indices,
                                  const Tensor& values,
                                  const TensorOptions& options,
                                  bool copy_ = false);

/// create a sparse tensor by warpping COO format data from outside.
/// There will be data copy if copy_ is true.
///
/// It will set 'coalesced_' to false by default, see NOTE [ To Coalesced ]
HICE_API Tensor wrap_sparse(ConstIntArrayRef dims, int32_t* indices_ptr,
                            void* values_ptr, const int64_t n_nonzero,
                            const TensorOptions& options, bool copy_ = false);

/// NOTE: wrap_sparse_unsafe is different from wrap_sparse.
/// See comment of new_sparse_unsafe
///
/// It will set 'coalesced_' to false by default, see NOTE [ To Coalesced ]
HICE_API Tensor wrap_sparse_unsafe(ConstIntArrayRef dims, int32_t* indices_ptr,
                                   void* values_ptr, const int64_t n_nonzero,
                                   const TensorOptions& options,
                                   bool copy_ = false);

/// NOTE [ Flatten COO Indices ]
// This helper function flattens a coo indices tensor (a Tensor) into
// a 1D indices tensor. E.g.,
//   input = [[2, 4, 0],
//            [3, 1, 10]]
//   dims = [2, 12]
//   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
//
// In other words, assuming that each `indices[i, :]` is a valid index to a
// tensor `t` of shape `dims`. This returns the corresponding indices
// to the flattened tensor `t.reshape( prod(dims[:indices.dim(0)]), -1
// )`. if force_clone is true, the result will forced to be a clone of self.
HICE_API Tensor flatten_indices(const Tensor& indices, ConstIntArrayRef dims,
                                bool force_clone = false);

// Flatten coo tensor's indices from nD to 1D, similar to NOTE [ Flatten
// COO Indices ], except this one allows partial flatten: only flatten on
// specified dims. Note that the flatten indices might be uncoalesced if
// dims_to_flatten.ndim() < sparse_dim. Also if input indices is already
// coalesced, the flattened indices will also be sorted.
//
// args:
//    indices: sparse tensor indices
//    dims: sparse tensor dims
//    dims_to_flatten: a list of dim index to flatten
//
// Ex1:
//   indices = [[2, 4, 0],
//             [3, 1, 3]]
//   dims = [2, 12]
//   dims_to_flatten = [0, 1]
//   new_indices = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 3 ] = [27, 49, 3]
//
// Ex2:
//   dims_to_flatten = [1]
//   new_indices = [ 3, 1, 3 ]  # uncoalesced
HICE_API Tensor
flatten_indices_by_dims(const Tensor& indices, const ConstIntArrayRef& dims,
                        const ConstIntArrayRef& dims_to_flatten);

// return a coalesced coo sparse
HICE_API Tensor dense_to_sparse(const Tensor& self);

// convert coo to dense
HICE_API Tensor sparse_to_dense(const Tensor& self);

// convert coo to csr
HICE_API Tensor sparse_to_csr(const Tensor& self);

/// NOTE: csr functions are only for developers

HICE_API const SparseTensorImplCSR& get_impl_csr(const Tensor& self);
HICE_API SparseTensorImplCSR& get_mutable_impl_csr(Tensor& self);
HICE_API Tensor new_csr(const TensorOptions& options);
HICE_API Tensor new_csr(ConstIntArrayRef dims, const TensorOptions& options);
HICE_API Tensor new_csr(ConstIntArrayRef dims, const Tensor& column_indices,
               const Tensor& row_offsets, const Tensor& values,
               const TensorOptions& options, bool copy_ = false);
HICE_API Tensor wrap_csr(ConstIntArrayRef dims, int32_t* column_indices_ptr,
                int32_t* row_offsets_ptr, void* values_ptr,
                const int64_t n_nonzero, const TensorOptions& options,
                bool copy_ = false);

//convert dense to csr
HICE_API Tensor dense_to_csr(const Tensor& self);
// convert csr to dense
HICE_API Tensor csr_to_dense(const Tensor& self);
// convert csr to coo
HICE_API Tensor csr_to_sparse(const Tensor& self);
}  // namespace hice
