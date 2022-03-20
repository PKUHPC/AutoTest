#include "hice/core/sparse_tensor_impl_csr.h"
#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/core/shape_util.h"

namespace hice {

SparseTensorImplCSR::SparseTensorImplCSR(const TensorOptions& options)
    : TensorImpl(options) {
  HICE_CHECK(options.layout_type() == kCSR)
      << "layout_type of TensorOptions must be Sparse csr";
  row_offsets_ = empty({0}, options.dtype(kInt32).layout(kDense));
  column_indices_ = empty({0}, options.dtype(kInt32).layout(kDense));
  values_ = empty({0}, options.layout(kDense));
  coalesced_ = false;
}

SparseTensorImplCSR::SparseTensorImplCSR(const TensorOptions& options,
                             const Tensor& column_indices,
                             const Tensor& row_offsets, const Tensor& values)
    : TensorImpl(options),
      column_indices_(column_indices),
      row_offsets_(row_offsets),
      values_(values) {
  HICE_CHECK(options.layout_type() == kCSR)
      << "layout_type of TensorOptions must be kCSR";
  HICE_CHECK(column_indices.scalar_type() == kInt32);
  HICE_CHECK(row_offsets.scalar_type() == kInt32);
  HICE_CHECK(options.scalar_type() == values.scalar_type());
  HICE_CHECK(column_indices.device_type() == values.device_type());
  HICE_CHECK(row_offsets.device_type() == values.device_type());
  HICE_CHECK(options.device_type() == values.device_type());
  HICE_CHECK(column_indices.is_dense());
  HICE_CHECK(row_offsets.is_dense());
  HICE_CHECK(values.is_dense());
  this->mutable_shape().mutable_layout().set_type(kCSR);
  HICE_CHECK(column_indices.ndim() == 1);
  HICE_CHECK(values.ndim() == 1);
  HICE_CHECK(values.dim(0) == column_indices.size());
}

int64_t SparseTensorImplCSR::stride(int64_t d) const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: stride";
}
std::vector<int64_t> SparseTensorImplCSR::strides() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: strides";
}
int64_t SparseTensorImplCSR::offset() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: offset";
}
void SparseTensorImplCSR::set_offset(int64_t offset) {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: set_offset";
}
const Storage& SparseTensorImplCSR::storage() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: storage";
}
Storage& SparseTensorImplCSR::mutable_storage() {
  HICE_LOG(ERROR)
      << "sparse tensor does not have member function: mutable_storage";
}
void SparseTensorImplCSR::set_storage(Storage storage) {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: set_storage";
}
bool SparseTensorImplCSR::is_default_layout() const {
  HICE_LOG(ERROR)
      << "sparse tensor does not have member function: is_default_layout";
}

bool SparseTensorImplCSR::has_storage() const { return false; }

void SparseTensorImplCSR::set_data_type(const DataType& dtype) {
  TensorImpl::set_data_type(dtype);
  values_.mutable_impl().set_data_type(dtype);
}

void SparseTensorImplCSR::set_column_indices_and_values_unsafe(const Tensor& row_offsets, const Tensor& column_indices,
                                                     const Tensor& values,
                                                     bool copy_) {
  HICE_CHECK(column_indices.scalar_type() == kInt32);
  HICE_CHECK(this->scalar_type() == values.scalar_type());
  HICE_CHECK(column_indices.device_type() == values.device_type());
  HICE_CHECK(this->device_type() == values.device_type());
  HICE_CHECK(column_indices.is_dense());
  HICE_CHECK(values.is_dense());
  HICE_CHECK(column_indices.ndim() == 1);
  HICE_CHECK(values.ndim() == 1);
  HICE_CHECK(values.dim(0) == column_indices.dim(0));
  if (copy_) {
    row_offsets_ = empty(row_offsets.dims(), row_offsets.options());
    column_indices_ = empty(column_indices.dims(), column_indices.options());
    values_ = empty(values.dims(), values.options());
    hice::copy(column_indices, column_indices_);
    hice::copy(values, values_);
  } else {
    row_offsets_ = row_offsets;
    column_indices_ = column_indices;
    values_ = values;
  }
  coalesced_ = false;
}

void SparseTensorImplCSR::update_coalesced() {
  std::cout << "csr update_coalesced is under developing" << '\n';
#if 0
  Tensor col_ind = column_indices_.device().is_cpu() ? column_indices_ : column_indices_.to(kCPU);
  int64_t nnz = n_nonzero();
  const int32_t* indices_ptr = ind.data<int32_t>();
  const int32_t* rowind = indices_ptr;
  const int32_t* colind = indices_ptr + nnz;
  bool coalesced = true;
  for (int i = 0; i < nnz - 1; ++i) {
    bool ordered = (rowind[i] < rowind[i + 1]) ||
                   (rowind[i] == rowind[i + 1] && colind[i] < colind[i + 1]);
    if (!ordered) {
      coalesced = false;
      break;
    }
  }
  coalesced_ = coalesced;
#endif
}

} // namespace hice
