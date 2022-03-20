#pragma once

#include "hice/core/tensor.h"
#include "hice/core/tensor_impl.h"

namespace hice {

/// Stored in COO format, indices + values.
class HICE_API SparseTensorImplCOO : public TensorImpl {
 public:
  /// Construct SparseTensorImplCOO by specifying the meta data.
  SparseTensorImplCOO(const TensorOptions& options);

  /// Construct SparseTensorImplCOO by specifying the meta data.
  /// This method does not allocate the memory(no element in the storage).
  SparseTensorImplCOO(const TensorOptions& options, const Tensor& indices,
                   const Tensor& values);

  const Tensor& indices() const { return indices_; }
  const Tensor& values() const { return values_; }
  Tensor& mutable_indices() { return indices_; }
  Tensor& mutable_values() { return values_; }
  int64_t n_nonzero() const { return values_.dim(0); }
  bool is_coalesced() const { return coalesced_; }
  void set_coalesced(bool coalesced) { coalesced_ = coalesced; }

  // Method to be disabled
  int64_t stride(int64_t d) const override;
  std::vector<int64_t> strides() const override;
  int64_t offset() const override;
  void set_offset(int64_t offset) override;
  const Storage& storage() const override;
  Storage& mutable_storage() override;
  void set_storage(Storage storage) override;
  bool is_default_layout() const override;
  // Method to be overwrite
  bool has_storage() const override;
  void set_data_type(const DataType& dtype) override;

  // Takes indices and values and directly puts them into the sparse tensor, no
  // copy. NOTE: this method is unsafe because it doesn't check whether any
  // indices are out of boundaries of 'shape', so it should ONLY be used where
  // we know that the indices are guaranteed to be within bounds.
  void set_indices_and_values_unsafe(const Tensor& indices,
                                     const Tensor& values, bool copy_ = false);

  void update_coalesced();

 private:
  Tensor indices_;  // [ndim, size_nonzeros]
  Tensor values_;   // [size_nonzeros]
  bool coalesced_ = false;
};

}  // namespace hice