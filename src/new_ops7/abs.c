#include "src/new_ops7/abs.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status abs_create_output(const Tensor input, Tensor* output) {
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

static Status abs_int_kernel(const Tensor input, Tensor* output) {
  Status status;
  int32_t* in_data = (int32_t*)aitisa_tensor_data(input);
  int32_t* out_data = (int32_t*)aitisa_tensor_data(*output);
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    out_data[i] = (int32_t)abs(in_data[i]);
  }
  return status;
}

static Status abs_double_kernel(const Tensor input, Tensor* output) {
  Status status;
  double* in_data = (double*)aitisa_tensor_data(input);
  double* out_data = (double*)aitisa_tensor_data(*output);
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    out_data[i] = (double)fabs(in_data[i]);
  }
  return status;
}

static Status abs_float_kernel(const Tensor input, Tensor* output) {
  Status status;
  float* in_data = (float*)aitisa_tensor_data(input);
  float* out_data = (float*)aitisa_tensor_data(*output);
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    out_data[i] = (float)fabsf(in_data[i]);
  }
  return status;
}

static Status abs_long_kernel(const Tensor input, Tensor* output) {
  Status status;
  int64_t* in_data = (int64_t*)aitisa_tensor_data(input);
  int64_t* out_data = (int64_t*)aitisa_tensor_data(*output);
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    out_data[i] = (int64_t)labs(in_data[i]);
  }
  return status;
}

Status aitisa_abs(const Tensor input, Tensor* output) {
  Status status = STATUS_SUCCESS;
  CHECK_STATUS(abs_create_output(input, output));

  DataType dtype = aitisa_tensor_data_type(input);
  switch ((dtype).code) {
    case TYPE_INT32:
      abs_int_kernel(input, output);
      break;
    case TYPE_INT64:
      abs_long_kernel(input, output);
      break;
    case TYPE_FLOAT:
      abs_float_kernel(input, output);
      break;
    case TYPE_DOUBLE:
      abs_double_kernel(input, output);
      break;
    default:
      return STATUS_TYPE_MISMATCH;
  }

  return status;
}