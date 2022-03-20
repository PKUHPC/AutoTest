#include <algorithm>
#include "hice/basic/factories.h"

namespace hice{

namespace {

void fill_dense(Tensor &tensor, Scalar fill_value, size_t begin, size_t end) {
  //std::cout << "In CPU fill dense" << std::endl;
  ScalarType scalar_type = tensor.scalar_type();
  HICE_DISPATCH_ALL_AND_COMPLEX_TYPES(scalar_type, "cpu_full_dense",
    [&] { 
       auto data_ptr = tensor.mutable_data<scalar_t>(); 
       auto typed_value = fill_value.to<scalar_t>();
       std::fill(data_ptr + begin, data_ptr + end, typed_value);
     } 
  );
}

} // anonymous namespace

HICE_REGISTER_KERNEL(fill_kernel_dispatcher, &fill_dense, {kCPU, kDense});

} // namespace hice

