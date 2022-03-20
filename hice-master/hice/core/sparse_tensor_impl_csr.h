#pragma once

#include "hice/core/tensor.h"
#include "hice/core/tensor_impl.h"

namespace hice {

/// NOTE: CSR is only for developers
class SparseTensorImplCSR : public TensorImpl {
 public:
  /// Construct SparseTensorImplCSR by specifying the meta data.
  /// This method does not allocate the memory(no element in the storage).
  SparseTensorImplCSR(const TensorOptions& options);
  SparseTensorImplCSR(const TensorOptions& options, const Tensor& column_indices,
                const Tensor& row_offsets, const Tensor& values);

  const Tensor& column_indices() const { return column_indices_; }
  const Tensor& row_offsets() const { return row_offsets_; }
  const Tensor& values() const { return values_; }
  Tensor& mutable_column_indices() { return column_indices_; }
  Tensor& mutable_row_offsets() { return row_offsets_; }
  Tensor& mutable_values() { return values_; }
  int64_t n_nonzero() const { return values_.dim(0); }
  bool is_coalesced() const { return coalesced_; }
  void set_coalesced(bool coalesced) { coalesced_ = coalesced; }
  void set_column_indices_and_values_unsafe(const Tensor& row_offsets, const Tensor& column_indices,
                                            const Tensor& values, bool copy_ = false);
  void update_coalesced();
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

 private:
  Tensor column_indices_;
  Tensor row_offsets_;
  Tensor values_;
  bool coalesced_ = false;
};

} // namespace hice
