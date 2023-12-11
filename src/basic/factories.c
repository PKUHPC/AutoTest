#include "src/basic/factories.h"

Status aitisa_full(DataType dtype, Device device, int64_t *dims, int64_t ndim,
                   double value, Tensor *output) {
  Tensor new_tensor;
  CHECK_STATUS(
      aitisa_create(dtype, device,  dims, ndim, NULL, 0, &new_tensor));
  int64_t size = aitisa_tensor_size(new_tensor);
  aitisa_castto_typed_value_func(dtype)(&value, &value);
  for (int i = 0; i < size; ++i) {
    aitisa_tensor_set_item(new_tensor, i, &value);
  }
  *output = new_tensor;
  return STATUS_SUCCESS;
}

double uniform_data(double a, double b,long int * seed) {
  double t;
  *seed = 2045.0 * (*seed) + 1;
  *seed = *seed - (*seed / 1048576) * 1048576;
  t = (*seed) / 1048576.0;
  t = a + (b - a) * t;
  return t;
}

Status aitisa_uniform(DataType dtype, Device device, int64_t *dims, int64_t ndim,
                      double a,double b,long int * seed, Tensor *output) {
  Tensor new_tensor;
  CHECK_STATUS(
      aitisa_create(dtype, device,  dims, ndim, NULL, 0, &new_tensor));
  int64_t size = aitisa_tensor_size(new_tensor);
  for (int i = 0; i < size; ++i) {
    double value = uniform_data(a,b,seed);
    aitisa_castto_typed_value_func(dtype)(&value, &value);
    aitisa_tensor_set_item(new_tensor, i, &value);
  }
  *output = new_tensor;
  return STATUS_SUCCESS;
}

