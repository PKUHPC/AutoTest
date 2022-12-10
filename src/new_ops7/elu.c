#include "src/new_ops7/elu.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status elu_create_output(const Tensor input, Tensor* output) {
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = kFloat;
  Device device = aitisa_tensor_device(input);
  status = aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor);
  *output = new_tensor;
  return status;
}

#define elu_kernel(typename, alpha)                                     \
  typename* in_data = aitisa_tensor_data(input);                        \
  float* out_data = aitisa_tensor_data(*output);                        \
  typename zero = 0;                                                    \
  typename one = 1;                                                     \
  int64_t size = aitisa_tensor_size(input);                             \
  for (int64_t i = 0; i < size; i++) {                                  \
    out_data[i] =                                                       \
        (float)((in_data[i] > zero) ? in_data[i]                        \
                                    : alpha * (exp(in_data[i]) - one)); \
  }

Status aitisa_elu(Tensor input, float alpha, Tensor* output) {
  Status status = STATUS_SUCCESS;
  CHECK_STATUS(elu_create_output(input, output));
  DataType dtype = aitisa_tensor_data_type(input);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, elu_kernel, alpha);
  return status;
}