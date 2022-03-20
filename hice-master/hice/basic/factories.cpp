#include "hice/basic/factories.h"
#include <algorithm>
#include "hice/core/shape_util.h"
#include "hice/util/copy_bytes.h"

namespace hice {

HICE_DEFINE_DISPATCHER(fill_kernel_dispatcher);
HICE_DEFINE_DISPATCHER(rand_uniform_kernel_dispatcher);
HICE_DEFINE_DISPATCHER(rand_normal_kernel_dispatcher);

// Create an empty dense tensor with allocating memory right away
#if 0
Tensor empty(ConstIntArrayRef dims, const TensorOptions& options) {
  if (options.layout_type() != kDense) {
    HICE_LOG(ERROR) << "The empty(...) is only implemented for dense storage";
  }
  auto tensor = Tensor(dims, options);
  tensor.raw_mutable_data();
  return tensor;
}
#else
Tensor empty(ConstIntArrayRef dims, const TensorOptions &options) {
  // construct Shape
  Shape shape;
  auto min2maj = options.layout().minor_to_major();
  if (min2maj.size() == 0) {
    shape = ShapeUtil::make_shape(dims);
  } else {
    shape = ShapeUtil::make_shape_with_layout(dims, min2maj);
  }

  Layout &layout = shape.mutable_layout();
  layout.set_type(options.layout().type());
  // construct Storage
  Allocator *allocator = get_allocator(options.device_type());
  int64_t size = ShapeUtil::get_num_items(shape);

  DataType dtype = options.data_type();
  Storage storage =
      make_storage<StorageImpl>(size, allocator->allocate(size * dtype.size()),
                                dtype, options.device(), allocator);
  Tensor tensor = make_tensor<TensorImpl>(shape, storage, 0);
  return tensor;
}
#endif

Tensor empty_like(const Tensor& self) {
  return empty(self.dims(), self.options());
}

Tensor full(ConstIntArrayRef dims, Scalar fill_value,
            const TensorOptions &options) {
  HICE_CHECK_SUPPORTED(options.layout_type() == kDense)
      << "Only implemented for dense storage";
  auto result = hice::empty(dims, options);
  fill_kernel_dispatcher(result, fill_value, 0, result.size());
  return result;
}

Tensor rand_uniform(ConstIntArrayRef dims, Scalar min, Scalar max,
                    const TensorOptions &options) {
  HICE_CHECK_SUPPORTED(options.layout_type() == kDense)
      << "Only implemented for dense storage";
  auto result = hice::empty(dims, options);
  rand_uniform_kernel_dispatcher(result, min, max);
  return result;
}

Tensor rand_normal(ConstIntArrayRef dims, Scalar mean, Scalar stddev,
                   const TensorOptions &options) {
  HICE_CHECK_SUPPORTED(options.layout_type() == kDense)
      << "Only implemented for dense storage";
  HICE_CHECK_SUPPORTED(options.scalar_type() == kDouble ||
                       options.scalar_type() == kFloat)
      << "Only implemented for double and float";
  auto result = hice::empty(dims, options);
  rand_normal_kernel_dispatcher(result, mean, stddev);
  return result;
}

Tensor create(ConstIntArrayRef dims, void *values_ptr, size_t len,
              const TensorOptions &options) {
  size_t item_size = options.data_type().size();
  size_t size_tensor = size_of_dims(dims);
  if (values_ptr == NULL) {
    return full(dims, 0, options);
  }
  // create an empty tensor and do copy
  Tensor tensor = empty(dims, options);
  size_t size_to_copy = std::min(len / item_size, size_tensor);
  auto device_type = tensor.device_type();
  void *data_ptr = tensor.raw_mutable_data();
  hice::copy_bytes(size_to_copy * item_size, values_ptr, device_type, data_ptr,
                   device_type);
  // set the rest to zero
  fill_kernel_dispatcher(tensor, 0, size_to_copy, size_tensor);
  return tensor;
}

Tensor wrap(ConstIntArrayRef dims, void *values_ptr,
            const TensorOptions &options, bool copy_) {
  Tensor tensor = Tensor(dims, options);
  if (copy_) {
    auto device_type = tensor.device_type();
    void *data_ptr = tensor.raw_mutable_data();
    hice::copy_bytes(tensor.size() * tensor.item_size(), values_ptr,
                     device_type, data_ptr, device_type);
  } else {
    Storage &storage = tensor.mutable_impl().mutable_storage();
    storage.set_data_ptr(DataPtr(const_cast<void *>(values_ptr),
                                 const_cast<void *>(values_ptr),
                                 &delete_nothing));
  }
  return tensor;
}

#ifdef HICE_USE_CUDA
// This function helps HICE to use the stream providing by users
void set_stream(cudaStream_t stream) { CUDAContext::set_stream(stream); }

void set_cublas_handle(cublasHandle_t handle) {
  CUDAContext::set_cublas_handle(handle);
}

void set_cusparse_handle(cusparseHandle_t handle) {
  CUDAContext::set_cusparse_handle(handle);
}

#ifdef HICE_USE_CUDNN
// This function helps HICE to use the cudnn handle providing by users
void set_cudnn_handle(cudnnHandle_t handle) {
  CUDAContext::set_cudnn_handle(handle);
}
#endif
#endif

}  // namespace hice
