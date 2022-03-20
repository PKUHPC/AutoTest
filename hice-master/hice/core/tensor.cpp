#include "hice/util/copy_bytes.h"
#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"
#include "hice/core/sparse_tensor.h"
#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/transpose.h"
#include "hice/basic/reshape.h"
#include "hice/basic/resize.h"
#include "hice/basic/copy.h"

namespace hice {

Tensor& Tensor::fill(Scalar value) {
  HICE_CHECK(is_dense()) << "fill only supports dense tensor";
  fill_kernel_dispatcher(*this, value, 0, this->size());
  return *this;
}

Tensor& Tensor::reshape(ConstIntArrayRef new_dims) {
  HICE_CHECK(is_dense()) << "reshape only supports dense tensor";
  return hice::reshape_(*this, new_dims);
}

Tensor& Tensor::expand_dims(int64_t axis) {
  HICE_CHECK(is_dense()) << "expand_dims only supports dense tensor";
  return hice::expand_dims_(*this, axis);
}

Tensor& Tensor::squeeze(int64_t axis) {
  HICE_CHECK(is_dense()) << "squeeze only supports dense tensor";
  return hice::squeeze_(*this, axis);
}

Tensor& Tensor::resize(ConstIntArrayRef new_dims) {
  return hice::resize_(*this, new_dims);
}

Tensor& Tensor::transpose(ConstIntArrayRef perm, bool conjugate) {
  return hice::transpose_(*this, perm, conjugate);
}

Tensor& Tensor::transpose_matrix(bool conjugate) {
  HICE_CHECK(is_dense()) << "transpose_matrix only supports dense tensor";
  return hice::transpose_matrix_(*this, conjugate);
}

Tensor Tensor::to(Device new_device) const {
  switch(layout_type()) {
    case kDense: {
      Tensor new_tensor = hice::empty(
          dims(), hice::device(new_device).layout(layout()).dtype(data_type()));
      hice::copy_bytes(storage().size() * item_size(), raw_data(), device(),
                       new_tensor.raw_mutable_data(), new_device);
      return new_tensor;
    }
    case kCOO: {
      Tensor indices_new = indices().to(new_device);
      Tensor values_new = values().to(new_device);
      Tensor new_tensor = new_sparse(dims(), indices_new, values_new, options().device(new_device));
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    case kCSR: {
      Tensor column_indices_new = column_indices().to(new_device);
      Tensor row_offsets_new = row_offsets().to(new_device);
      Tensor values_new = values().to(new_device);
      Tensor new_tensor = new_csr(dims(), column_indices_new, row_offsets_new,
                                  values_new, options().device(new_device));
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    default:
      HICE_LOG(ERROR) << "Tensor::to(Device new_device) Unsupported storage layout";
  }
}

Tensor Tensor::to(ScalarType stype) const {
  switch(layout_type()) {
    case kDense: {
      Tensor dst(this->dims(),
                hice::device(this->device()).dtype(stype).layout(this->layout()));
      hice::copy(*this, dst);
      return dst;
    }
    case kCOO: {
      Tensor indices_new = indices().clone();
      Tensor values_new = values().to(stype);
      Tensor new_tensor = new_sparse(dims(), indices_new, values_new, options().dtype(stype));
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    case kCSR: {
      Tensor column_indices_new = column_indices().clone();
      Tensor row_offsets_new = row_offsets().clone();
      Tensor values_new = values().to(stype);
      Tensor new_tensor = new_csr(dims(), column_indices_new, row_offsets_new,
                                  values_new, options().dtype(stype));
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    default:
      HICE_LOG(ERROR) << "Tensor::to(ScalarType stype) Unsupported storage layout";
  }
}

Tensor Tensor::to(LayoutType ltype) const {
  if (layout_type() == ltype) {
    return clone();
  } else if (is_dense() && ltype == kCOO) {
    return dense_to_sparse(*this);
  } else if (is_coo() && ltype == kDense) {
    return sparse_to_dense(*this);
  } else if (is_coo() && ltype == kCSR) {
    return sparse_to_csr(*this);
  } else if (is_dense() && ltype == kCSR) {
    return dense_to_csr(*this);
  } else if (is_csr() && ltype == kDense) {
    return csr_to_dense(*this);
  } else if (is_csr() && ltype == kCOO) {
    return csr_to_sparse(*this);
  } else {
    HICE_LOG(ERROR) << "Tensor::to(LayoutType ltype) Unsupported storage layout";
  }
}

Tensor Tensor::clone() const {
  switch(layout_type()) {
    case kDense: {
      Tensor dst(this->dims(), this->options());
      hice::copy(*this, dst);
      return dst;
    }
    case kCOO: {
      Tensor indices_new = indices().clone();
      Tensor values_new = values().clone();
      Tensor new_tensor = new_sparse(dims(), indices_new, values_new, options());
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    case kCSR: {
      Tensor row_offsets_new = row_offsets().clone();
      Tensor column_indices_new = column_indices().clone();
      Tensor values_new = values().clone();
      Tensor new_tensor = new_csr(dims(), column_indices_new, row_offsets_new,
                                  values_new, options());
      new_tensor.set_coalesced(this->is_coalesced());
      return new_tensor;
    }
    default:
      HICE_LOG(ERROR) << "Tensor::clone() Unsupported storage layout";
  }
}

const Tensor& Tensor::indices() const {
  HICE_CHECK(is_coo()) << "indices: not a coo sparse tensor";
  return get_impl_coo(*this).indices();
}
Tensor& Tensor::mutable_indices() {
  HICE_CHECK(is_coo()) << "mutable_indices: not a coo sparse tensor";
  return get_mutable_impl_coo(*this).mutable_indices();
}
bool Tensor::is_coalesced() const {
  HICE_CHECK(is_coo()||is_csr()) << "is_coalesced: not a coo or csr sparse tensor";
  if (is_coo())
    return get_impl_coo(*this).is_coalesced();
  else {
    return get_impl_csr(*this).is_coalesced();
  }
}
void Tensor::set_coalesced(bool coalesced) {
  HICE_CHECK(is_coo()||is_csr()) << "is_coalesced: not a coo or csr sparse tensor";
  if (is_coo())
    get_mutable_impl_coo(*this).set_coalesced(coalesced);
  else {
    get_mutable_impl_csr(*this).set_coalesced(coalesced);
  }
}

Tensor& Tensor::resize_with_nnz(int64_t new_n_nonzero) {
  HICE_CHECK((is_coo() || is_csr())) << "resize_with_nnz: not a coo or csr sparse tensor";
  Tensor& values = mutable_values();
  values.resize({new_n_nonzero});
  int64_t ndim = this->ndim();
  if (is_coo()) {
    Tensor& indices = mutable_indices();
    if (n_nonzero() > 0) {
      indices.transpose_matrix();
      contiguous_(indices);
      indices.resize({new_n_nonzero, ndim});
      indices.transpose_matrix();
      contiguous_(indices);
    } else {
      indices.resize({ndim, new_n_nonzero});
    }
  } else {
    Tensor& column_indices = mutable_column_indices();
    column_indices.resize({new_n_nonzero});
  }
}

Tensor Tensor::to_coalesced() const {
  HICE_CHECK(is_coo()||is_csr()) << "order_to_coalesced: not a coo or csr tensor";
  HICE_CHECK(!is_empty());
  if (is_coalesced()) {
    return *this;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  int64_t nnz = n_nonzero();
  if (nnz < 2) {
    Tensor dst = this->clone();
    dst.set_coalesced(true);
    return dst;
  }
  if(is_coo()){
    Tensor self = this->device().is_cpu() ? (*this) : this->to(kCPU);
    const Tensor& indices = self.indices();
    const Tensor& values = self.values();
    int64_t ndim = self.ndim();
    const int32_t* indices_data = indices.data<int32_t>();
    const int32_t* row_ind_data = indices_data;
    const int32_t* col_ind_data = indices_data + nnz;
    // set permutation
    std::vector<int64_t> perm_indices(nnz, 0);
    for (int i = 0; i < perm_indices.size(); ++i) {
      perm_indices[i] = i;
    }
    // ascending
    std::sort(perm_indices.begin(), perm_indices.end(),
      [=](int64_t i1, int64_t i2) {
        return (row_ind_data[i1] < row_ind_data[i2]) ||
                (row_ind_data[i1] == row_ind_data[i2] &&
                col_ind_data[i1] <= col_ind_data[i2]);
      }
    );
    // get new nnz
    int64_t nnz_coalesce = nnz;
    for (int i = 0; i < nnz - 1; ++i) {
      int cur = perm_indices[i];
      int next = perm_indices[i + 1]; // we have checked nnz >= 2, so it will not be out bound
      if (row_ind_data[cur] == row_ind_data[next] &&
          col_ind_data[cur] == col_ind_data[next]) {
        nnz_coalesce -= 1;
      }
    }
    // allocate indices and values
    Tensor indices_coalesce = empty({ndim, nnz_coalesce}, indices.options());
    Tensor values_coalesce = empty({nnz_coalesce}, values.options());
    // set indices and values
    HICE_DISPATCH_ALL_TYPES(values_coalesce.scalar_type(), "coalesce", [&]() {
      const scalar_t* values_data = values.data<scalar_t>();
      scalar_t* values_coal_data = values_coalesce.mutable_data<scalar_t>();
      int32_t* indices_coal_data = indices_coalesce.mutable_data<int32_t>();
      int32_t* row_ind_coal_data = indices_coal_data;
      int32_t* col_ind_coal_data = indices_coal_data + nnz_coalesce;
      // init
      row_ind_coal_data[0] = row_ind_data[perm_indices[0]];
      col_ind_coal_data[0] = col_ind_data[perm_indices[0]];
      values_coal_data[0] = values_data[perm_indices[0]];
      // process
      int idx_ind_coal = 0;
      for (int i = 1; i < nnz; ++i) {
        int cur = perm_indices[i];
        if (row_ind_data[cur] == row_ind_coal_data[idx_ind_coal] &&
            col_ind_data[cur] == col_ind_coal_data[idx_ind_coal]) {
          values_coal_data[idx_ind_coal] += values_data[cur];
        } else {
          idx_ind_coal += 1;
          row_ind_coal_data[idx_ind_coal] = row_ind_data[cur];
          col_ind_coal_data[idx_ind_coal] = col_ind_data[cur];
          values_coal_data[idx_ind_coal] = values_data[cur];
        }
      }
    });
    Tensor dst = new_sparse(dims(), self.options());
    SparseTensorImplCOO& spimpl = get_mutable_impl_coo(dst);
    spimpl.set_indices_and_values_unsafe(indices_coalesce, values_coalesce);
    spimpl.set_coalesced(true);
    if (device().is_cpu()) {
      return dst;
    } else {
      return dst.to(device());
    }
  } else {    //csr
    Tensor self = this->device().is_cpu() ? (*this) : this->to(kCPU); //csr not supported
    const Tensor& column_indices = self.column_indices();
    const Tensor& row_offsets = self.row_offsets();
    const Tensor& values = self.values();
    int64_t ndim = self.ndim();
    const int32_t* column_indices_data = column_indices.data<int32_t>();
    const int32_t* row_offsets_data = row_offsets.data<int32_t>();
    // set permutation
    std::vector<int64_t> perm_indices(nnz, 0);
    for (int i = 0; i < perm_indices.size(); ++i) {
      perm_indices[i] = i;
    }
    // ascending
    for(int i = 0; i < row_offsets.size() - 1; i++){
      int row_start = row_offsets_data[i];
      int row_end = row_offsets_data[i + 1];
      std::sort(perm_indices.begin() + row_start, perm_indices.begin() + row_end,
        [=](int64_t i1, int64_t i2) {
          return column_indices_data[i1] <= column_indices_data[i2];
        }
      );
    }
    // get new nnz
    int64_t nnz_coalesce = nnz;          // csr does not permit duplicate entries
    Tensor column_indices_coalesce = empty({nnz_coalesce}, column_indices.options());
    Tensor values_coalesce = empty({nnz_coalesce}, values.options());
    // set indices and values
    HICE_DISPATCH_ALL_TYPES(values_coalesce.scalar_type(), "coalesce", [&]() {
      const scalar_t* values_data = values.data<scalar_t>();
      scalar_t* values_coal_data = values_coalesce.mutable_data<scalar_t>();
      int32_t* column_indices_coal_data = column_indices_coalesce.mutable_data<int32_t>();
      // init
      column_indices_coal_data[0] = column_indices_data[perm_indices[0]];
      values_coal_data[0] = values_data[perm_indices[0]];
      // process
      for(int i = 0; i < row_offsets.size() - 1; i++) {
        int row_start = row_offsets_data[i];
        int row_end = row_offsets_data[i + 1];
        for(int j = row_start; j < row_end; j++) {
          int cur = perm_indices[j];
          column_indices_coal_data[j] = column_indices_data[cur];
          values_coal_data[j] = values_data[cur];
        }
      }
    });
    Tensor dst = new_csr(dims(), self.options());
    SparseTensorImplCSR& csr_impl = get_mutable_impl_csr(dst);
    csr_impl.set_column_indices_and_values_unsafe(row_offsets, column_indices_coalesce, values_coalesce);
    csr_impl.set_coalesced(true);
    if (device().is_cpu()) {
      return dst;
    } else {
      return dst.to(device());
    }
  }
}

void Tensor::update_coalesced() {
  HICE_CHECK(is_coo()) << "update_coalesced: not a coo sparse tensor";
  get_mutable_impl_coo(*this).update_coalesced();
}

const Tensor& Tensor::column_indices() const {
  HICE_CHECK(is_csr()) << "column_indices: not a csr sparse tensor";
  return get_impl_csr(*this).column_indices();
}
Tensor& Tensor::mutable_column_indices() {
  HICE_CHECK(is_csr()) << "mutable_column_indices: not a csr sparse tensor";
  return get_mutable_impl_csr(*this).mutable_column_indices();
}
const Tensor& Tensor::row_offsets() const {
  HICE_CHECK(is_csr()) << "row_offsets: not a csr sparse tensor";
  return get_impl_csr(*this).row_offsets();
}
Tensor& Tensor::mutable_row_offsets() {
  HICE_CHECK(is_csr()) << "mutable_row_offsets: not a csr sparse tensor";
  return get_mutable_impl_csr(*this).mutable_row_offsets();
}

int64_t Tensor::n_nonzero() const {
  if (is_coo()) {
    return get_impl_coo(*this).n_nonzero();
  } else if (is_csr()) {
    return get_impl_csr(*this).n_nonzero();
  }
  HICE_LOG(ERROR) << "n_nonzero: not a sparse tensor";
}
const Tensor& Tensor::values() const {
  if (is_coo()) {
    return get_impl_coo(*this).values();
  } else if (is_csr()) {
    return get_impl_csr(*this).values();
  }
  HICE_LOG(ERROR) << "values: not a sparse tensor";
}
Tensor& Tensor::mutable_values() {
  if (is_coo()) {
    return get_mutable_impl_coo(*this).mutable_values();
  } else if (is_csr()) {
    return get_mutable_impl_csr(*this).mutable_values();
  }
  HICE_LOG(ERROR) << "values: not a sparse tensor";
}


} // namespace hice
