#include "hice/basic/resize.h"
#include "hice/basic/memset.h"
#include "hice/core/shape_util.h"
#include "hice/core/sparse_tensor.h"

namespace hice {

void resize_storage(Storage& storage, ptrdiff_t new_size) {
  DataPtr new_data;
  size_t item_size = storage.data_type().size();
  if (new_size != 0) {
    new_data =
        storage.allocator()->allocate(item_size * new_size);
  }
  DataPtr old_data = storage.set_data_ptr(std::move(new_data));
  ptrdiff_t old_size = storage.size();
  storage.set_size(new_size);
  if (old_data != nullptr) {
    ptrdiff_t copy_size = old_size;
    if (storage.size() < copy_size) {
      copy_size = storage.size();
    }
    if (copy_size > 0) {
      hice::memset(storage.raw_data(), 0, storage.size() * item_size,
                   storage.device());
      hice::copy_bytes(copy_size * item_size, old_data.get(), storage.device(),
                       storage.raw_data(), storage.device());
    }
  }
}

Tensor& resize_dense(Tensor& self, ConstIntArrayRef new_dims) {
    int64_t new_size = hice::size_of_dims(new_dims);
    // Step 1: adjust the shape for the new_dims
    TensorImpl &impl = self.mutable_impl();
    if (impl.ndim() == new_dims.size()) {
      // No need to change the layout if the new_dims' size equals the old one
      impl.set_shape(ShapeUtil::make_shape_with_layout(
          new_dims, impl.shape().layout().minor_to_major()));
    } else {
      if (impl.is_default_layout()) {
        // Make a new shape with the default layout if the layout of the old shape
        // is a default one
        impl.set_shape(ShapeUtil::make_shape(new_dims));
      } else {
        // Make a new shape with a layout aligned to the old one to make sure not
        // rearrange the underlying storage
        auto new_shape = ShapeUtil::align_layouts(
            impl.shape(), ShapeUtil::make_shape(new_dims));
        if(new_shape) {
          impl.set_shape(new_shape.value());
        } else {
          // Todo: rearrange the underlying storage by cloning the tensor
          HICE_CHECK(false) << "Cannot set dimensions without rearranging the "
                            "underlying storage to the default layout";
          // // make the physical data contiguous
          // Tensor tensor_new(self.dims(), device(self.device()).dtype(self.data_type()));
          // copy(self, tensor_new);
          // impl.set_storage(tensor_new.storage());
          // impl.set_shape(ShapeUtil::make_shape(new_dims));
        }
      }
    }
    // Step 2: resize the underlying storage if necessary
    // FIXME: lazy initialized should be tackle
    // if (impl.storage_initialized()) {
      if (self.offset() + new_size > impl.storage().size()) {
        Storage& storage = impl.mutable_storage();
        resize_storage(storage, self.offset() + new_size);
      }
    // }
    return self;
}

// NOTE: This function preserves invariants of dimensions with respect to
// indices and values.
// NOTE: This function supports the following cases:
// 1. When we keep the number of dimensions unchanged, and NOT shrinking the
// size of any of the dimensions.
// 2. When the sparse tensor has zero non-zero entities, in which case we are
// free to change the shape.
//
// This function DOESN'T support (and will throw an error) the following
// cases:
// 1. When we attempt to change the number of dimensions on a non-empty sparse
// tensor (such an operation will invalidate the indices stored).
// 2. When we attempt to shrink the size of any of the sparse dimensions on a
// non-empty sparse tensor (this could make some of the stored indices
// out-of-bound and thus unsafe).
Tensor& resize_sparse(Tensor& self, ConstIntArrayRef new_dims) {
  auto dims = self.dims();
  auto ndim = self.ndim();
  if (self.n_nonzero() > 0) {
    HICE_CHECK(ndim == static_cast<int64_t>(new_dims.size()));
    bool shrinking_dims = false;
    for (int64_t i = 0; i < ndim; i++) {
      if (new_dims[i] < dims[i]) {
        shrinking_dims = true;
        break;
      }
    }
    HICE_CHECK(!shrinking_dims)
        << "shrinking the size of dimensions on a non-empty sparse "
           "tensor is not supported.";
  } else if (self.n_nonzero() == 0) {
    SparseTensorImplCOO& spimpl = get_mutable_impl_coo(self);
    Tensor& indices = spimpl.mutable_indices();
    resize_dense(indices, {(int64_t)new_dims.size(), 0});
  }
  if (new_dims != dims) {
    // set shape
    auto new_shape = ShapeUtil::make_shape(new_dims);
    new_shape.mutable_layout().set_type(kCOO);
    SparseTensorImplCOO& spimpl = get_mutable_impl_coo(self);
    spimpl.set_shape(new_shape);
  }
}
Tensor& resize_csr(Tensor& self, ConstIntArrayRef new_dims) {
  auto dims = self.dims();
  auto ndim = self.ndim();
  int64_t new_ndim = new_dims.size();
  HICE_CHECK(new_ndim <= 2)
      << "Can not resize a csr tensor with over 2 dimensions";
  int64_t new_n_rows = new_dims[0];
  if (self.n_nonzero() > 0) {
    HICE_CHECK(ndim == static_cast<int64_t>(new_dims.size()));
    bool shrinking_dims = false;
    for (int64_t i = 0; i < ndim; i++) {
      if (new_dims[i] < dims[i]) {
        shrinking_dims = true;
        break;
      }
    }
    HICE_CHECK(!shrinking_dims)
        << "shrinking the size of dimensions on a non-empty csr "
           "tensor is not supported.";
  } else if (self.n_nonzero() == 0) {
    SparseTensorImplCSR& csr_impl = get_mutable_impl_csr(self);
    Tensor& row_offsets = csr_impl.mutable_row_offsets();
    resize_dense(row_offsets, {new_n_rows + 1});
  }
  if (new_dims != dims) {
    // set shape
    auto new_shape = ShapeUtil::make_shape(new_dims);
    new_shape.mutable_layout().set_type(kCSR);
    SparseTensorImplCSR& csr_impl = get_mutable_impl_csr(self);
    csr_impl.set_shape(new_shape);
  }
}
Tensor& resize_(Tensor& self, ConstIntArrayRef new_dims) {
  switch(self.layout_type()) {
    case kDense: {
      return resize_dense(self, new_dims);
    }
    case kCOO: {
      return resize_sparse(self, new_dims);
    }
    case kCSR: {
      return resize_csr(self, new_dims);
    }
    default:
      HICE_LOG(ERROR) << "resize_: Unsupported storage layout";
  }
}

} // namespace hice
