#include "src/new_ops7/square.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status square_create_output(const Tensor input, Tensor* output) {
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  status = aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor);
  *output = new_tensor;
  return status;
}

#define square_kernel(typename)                     \
  typename* in_data = aitisa_tensor_data(input);    \
  typename* out_data = aitisa_tensor_data(*output); \
  int64_t size = aitisa_tensor_size(input);         \
  for (int64_t i = 0; i < size; i++) {              \
    out_data[i] = (typename)pow(in_data[i], 2);     \
  }

Status aitisa_square(const Tensor input, Tensor* output) {
  Status status = STATUS_SUCCESS;
  CHECK_STATUS(square_create_output(input, output));
  DataType dtype = aitisa_tensor_data_type(input);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, square_kernel);
  return status;
}