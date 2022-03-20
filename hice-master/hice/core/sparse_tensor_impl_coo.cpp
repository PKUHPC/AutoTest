#include "hice/core/sparse_tensor_impl_coo.h"
#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/core/shape_util.h"

namespace hice {

// An empty dense tensor defaults to a 0-dimensional tensor. Thus, an empty
// sparse tensor should be a 0-dimensional tensor of size [].
// This means that we allocate a [0,0] size indices tensor and a [0] size
// values tensor for such an empty sparse tensor.
SparseTensorImplCOO::SparseTensorImplCOO(const TensorOptions& options)
    : TensorImpl(options) {
  HICE_CHECK(options.layout_type() == kCOO)
      << "layout_type of TensorOptions must be Sparse";
  indices_ = empty({0, 0}, options.dtype(kInt32).layout(kDense));
  values_ = empty({0}, options.layout(kDense));
  coalesced_ = false;
}

// TensorImpl has no default constructor, so base constructor
// with parameters must be called in the initialization list.
// Besides, sparse tensor need storage to keep informations about device_type
// and data_type, meanwhile there should be no data memory allocated in the
// storage. so TensorImpl(const TensorOptions& options) is called to init shape
// and storage(size=0), then consistency check and would be done.
// NOTE: scalar tensor must be dense
SparseTensorImplCOO::SparseTensorImplCOO(const TensorOptions& options,
                                   const Tensor& indices, const Tensor& values)
    : TensorImpl(options), indices_(indices), values_(values) {
  HICE_CHECK(options.layout_type() == kCOO)
      << "layout_type of TensorOptions must be kCOO";
  HICE_CHECK(indices.scalar_type() == kInt32);
  HICE_CHECK(options.scalar_type() == values.scalar_type());
  HICE_CHECK(indices.device_type() == values.device_type());
  HICE_CHECK(options.device_type() == values.device_type());
  HICE_CHECK(indices.is_dense());
  HICE_CHECK(values.is_dense());
  this->mutable_shape().mutable_layout().set_type(kCOO);

  // Only indices/values of the following shape are allowed. ( empty coo)
  HICE_CHECK_EQ(compare_dims(indices.dims(), {0, 0}), 0);
  HICE_CHECK_EQ(compare_dims(values.dims(), {0}), 0);
  coalesced_ = false;
}

int64_t SparseTensorImplCOO::stride(int64_t d) const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: stride";
}
std::vector<int64_t> SparseTensorImplCOO::strides() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: strides";
}
int64_t SparseTensorImplCOO::offset() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: offset";
}
void SparseTensorImplCOO::set_offset(int64_t offset) {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: set_offset";
}
const Storage& SparseTensorImplCOO::storage() const {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: storage";
}
Storage& SparseTensorImplCOO::mutable_storage() {
  HICE_LOG(ERROR)
      << "sparse tensor does not have member function: mutable_storage";
}
void SparseTensorImplCOO::set_storage(Storage storage) {
  HICE_LOG(ERROR) << "sparse tensor does not have member function: set_storage";
}
bool SparseTensorImplCOO::is_default_layout() const {
  HICE_LOG(ERROR)
      << "sparse tensor does not have member function: is_default_layout";
}

bool SparseTensorImplCOO::has_storage() const { return false; }

void SparseTensorImplCOO::set_data_type(const DataType& dtype) {
  TensorImpl::set_data_type(dtype);
  values_.mutable_impl().set_data_type(dtype);
}

void SparseTensorImplCOO::set_indices_and_values_unsafe(const Tensor& indices,
                                                     const Tensor& values,
                                                     bool copy_) {
  HICE_CHECK(indices.scalar_type() == kInt32);
  HICE_CHECK(this->scalar_type() == values.scalar_type());
  HICE_CHECK(indices.device_type() == values.device_type());
  HICE_CHECK(this->device_type() == values.device_type());
  HICE_CHECK(indices.is_dense());
  HICE_CHECK(values.is_dense());
  HICE_CHECK(indices.ndim() == 2);
  HICE_CHECK(values.ndim() == 1);
  HICE_CHECK(values.dim(0) == indices.dim(1));
  if (copy_) {
    indices_ = empty(indices.dims(), indices.options());
    values_ = empty(values.dims(), values.options());
    hice::copy(indices, indices_);
    hice::copy(values, values_);
  } else {
    indices_ = indices;
    values_ = values;
  }
  coalesced_ = false;
}

void SparseTensorImplCOO::update_coalesced() {
  Tensor ind = indices_.device().is_cpu() ? indices_ : indices_.to(kCPU);
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
}

}  // namespace hice