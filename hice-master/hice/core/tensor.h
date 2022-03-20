#pragma once

#include "hice/core/tensor_impl.h"

namespace hice {

class HICE_API Tensor final {
 private:
  using TensorImplPtr = intrusive_ptr<TensorImpl>;

 public:
  Tensor() = default;

  explicit Tensor(TensorOptions options)
      : pimpl_(make_intrusive<TensorImpl>(options)) {}

  explicit Tensor(ConstIntArrayRef dims,
                  TensorOptions options = TensorOptions())
      : pimpl_(make_intrusive<TensorImpl>(dims, options)) {}

  // This constructor only used by HICE developers
  explicit Tensor(TensorImplPtr pimpl) : pimpl_(std::move(pimpl)) {
    if (pimpl_.get() == nullptr) {
      throw std::runtime_error("TensorImpl with nullptr is not supported");
    }
  }

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  int64_t size() const { return pimpl_->size(); }
  int64_t ndim() const { return pimpl_->ndim(); }
  int64_t dim(int64_t d) const { return pimpl_->dim(get_true_axis(d)); }
  ConstIntArrayRef dims() const { return pimpl_->dims(); }
  int64_t stride(int64_t d) const { return pimpl_->stride(get_true_axis(d)); }
  std::vector<int64_t> strides() const { return pimpl_->strides(); }
  int64_t offset() const { return pimpl_->offset(); }

  DeviceType device_type() const { return pimpl_->device_type(); }
  Device device() const { return pimpl_->device(); }
  ScalarType scalar_type() const { return pimpl_->scalar_type(); }
  const DataType& data_type() const { return pimpl_->data_type(); }

  const Shape& shape() const { return pimpl_->shape(); }
  LayoutType layout_type() const { return pimpl_->layout_type(); }
  const Layout& layout() const { return pimpl_->layout(); }

  const Storage& storage() const { return pimpl_->storage(); }
  bool has_storage() const { return pimpl_->has_storage(); }

  TensorOptions options() const {
    return TensorOptions().dtype(data_type()).device(device()).layout(layout());
  }
  size_t item_size() const { return pimpl_->item_size(); }

  template <typename T>
  const T* data() const {
    return pimpl_->data<T>();
  }
  const void* raw_data() const { return pimpl_->raw_data(); }
  template <typename T>
  T* mutable_data() {
    return pimpl_->mutable_data<T>();
  }
  void* raw_mutable_data() { return pimpl_->raw_mutable_data(); }

  const TensorImpl& impl() const { return *pimpl_.get(); }
  TensorImpl& mutable_impl() { return *pimpl_.get(); }

  bool is_same(const Tensor& other) const { return pimpl_ == other.pimpl_; }

  bool is_empty() const { return pimpl_->is_empty(); }
  bool is_dense() const { return pimpl_->is_dense(); }
  bool is_coo() const { return pimpl_->is_coo(); }
  bool is_csr() const { return pimpl_->is_csr(); }
  bool is_sparse() const { return pimpl_->is_csr() || pimpl_->is_coo(); }
  bool is_default_layout() const { return pimpl_->is_default_layout(); }
  int type_id() const {
    return static_cast<int>(device_type()) +
           kNumDeviceTypes * static_cast<int>(layout_type());
  }
  bool is_defined() const { return pimpl_; }

  int64_t size_from_dim(int64_t k) const {
    return hice::size_from_dim(k, pimpl_->dims());
  }
  int64_t size_to_dim(int64_t k) const {
    return hice::size_to_dim(k, pimpl_->dims());
  }
  int64_t size_between_dim(int64_t k, int64_t l) const {
    return hice::size_between_dim(k, l, pimpl_->dims());
  }
  int64_t get_true_axis(int64_t axis) const {
    return hice::get_true_axis(axis, pimpl_->ndim());
  }

  // Tensor Methods
  Tensor& reshape(ConstIntArrayRef new_dims);
  Tensor& expand_dims(int64_t axis);
  Tensor& squeeze(int64_t axis);
  Tensor& resize(ConstIntArrayRef new_dims);
  Tensor& transpose(ConstIntArrayRef perm, bool conjugate = false);
  Tensor& transpose_matrix(bool conjugate = false);
  Tensor& fill(Scalar value);
  Tensor to(Device device) const;
  Tensor to(ScalarType stype) const;
  Tensor to(LayoutType ltype) const;
  Tensor clone() const;

  /// API: Sparse module(COO)
  const Tensor& indices() const;
  Tensor& mutable_indices();
  bool is_coalesced() const;
  void set_coalesced(bool coalesced);
  /// NOTE: This function preserves invariants of dimensions with respect to
  /// indices and values.
  Tensor& resize_with_nnz(int64_t new_n_nonzero);
  /// NOTE: [ To Coalesced ]
  /// return a new tensor with coalesced indices and values
  /// This method is very expensive. If you can guarante, you can
  /// set 'coalesced_' to true using tensor.set_coalesced().
  Tensor to_coalesced() const;
  // check and reset coalesced_. There will be data_copy from gpu to cpu when
  // indices is on a GPU device
  void update_coalesced();

  /// API: CSR
  const Tensor& column_indices() const;
  Tensor& mutable_column_indices();
  const Tensor& row_offsets() const;
  Tensor& mutable_row_offsets();

  /// API: CSR && COO
  int64_t n_nonzero() const;
  const Tensor& values() const;
  Tensor& mutable_values();

 private:
  TensorImplPtr pimpl_;
};

template <class T>
struct is_tensor
    : std::integral_constant<
          bool, std::is_same<Tensor, typename std::remove_cv<T>::type>::value> {
};

template <typename T, typename... Args>
Tensor make_tensor(Args&&... args) {
  return Tensor(hice::make_intrusive<T>(std::forward<Args>(args)...));
}

}  // namespace hice
