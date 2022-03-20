#include <algorithm>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "hice/basic/factories.h"

namespace hice{

namespace {

void fill_dense(Tensor &tensor, Scalar fill_value, size_t begin, size_t end) {
  ScalarType scalar_type = tensor.scalar_type();
  HICE_DISPATCH_ALL_AND_COMPLEX_TYPES(scalar_type, "cuda_fill_dense",
    [&] {
      auto *data_ptr = tensor.mutable_data<scalar_t>();
      thrust::device_ptr<scalar_t> dev_ptr(data_ptr);
      auto typed_value = fill_value.to<scalar_t>();
      thrust::fill(dev_ptr + begin, dev_ptr + end, typed_value);
     }
  );
}

} // anonymous namespace

HICE_REGISTER_KERNEL(fill_kernel_dispatcher, &fill_dense, {kCUDA, kDense});

}
