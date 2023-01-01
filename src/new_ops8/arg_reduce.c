#include "src/new_ops8/arg_reduce.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include "src/basic/factories.h"
#include "src/basic/index_utils.h"
#include "src/basic/squeeze.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status arg_create_output(const Tensor input, const int64_t dim,
                                Tensor* output) {
  Status status;
  int64_t* input_dims = aitisa_tensor_dims(input);
  int64_t input_ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = kInt64;
  Device device = aitisa_tensor_device(input);
  int64_t new_tensor_dims[input_ndim];
  memcpy(new_tensor_dims, input_dims, sizeof(int64_t) * input_ndim);
  new_tensor_dims[dim] = 1;
  status =
      aitisa_full(dtype, device, new_tensor_dims, input_ndim, 0, &new_tensor);
  *output = new_tensor;
  return status;
}

#define arg_kernel(typename, greater)                           \
  typename* in_data = (typename*)aitisa_tensor_data(input);     \
  double* out_data = (double*)aitisa_tensor_data(*output);      \
  int64_t n = input_dims[dim];                                  \
  int64_t stride = input_stride[dim];                           \
  if (n == 1) {                                                 \
    stride = 1;                                                 \
    for (int64_t i = input_ndim - 1; i > dim; i--) {            \
      stride *= input_dims[i];                                  \
    }                                                           \
  }                                                             \
  int64_t batch = num_items / (n * stride);                     \
  for (int64_t index = 0; index < batch * stride; index++) {    \
    int64_t b = index / stride;                                 \
    int64_t i = index % stride;                                 \
    typename* data = &in_data[b * n * stride + i];              \
    typename result = data[0];                                  \
    int64_t result_index = 0;                                   \
    for (int64_t k = 0; k < n; k++) {                           \
      typename value = data[k * stride];                        \
      bool cmp = greater ? (result > value) : (result < value); \
      result = cmp ? result : value;                            \
      result_index = cmp ? result_index : k;                    \
      if (!result) {                                            \
        break;                                                  \
      }                                                         \
    }                                                           \
    out_data[b * stride + i] = result_index;                    \
  }

Status aitisa_argmax(const Tensor input, const int64_t dim, const int keepdim,
                     Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  CHECK_STATUS(arg_create_output(input, dim, output));

  int64_t num_items = aitisa_tensor_size(input);
  int64_t input_ndim = aitisa_tensor_ndim(input);
  int64_t* input_dims = aitisa_tensor_dims(input);

  int64_t input_stride[input_ndim];
  aitisa_get_all_strides(input, input_stride);

  bool greater = true;
  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, arg_kernel, greater);
  if (!keepdim) {
    aitisa_squeeze(*output, NULL, 0, output);
  }
  return status;
}

Status aitisa_argmin(const Tensor input, const int64_t dim, const int keepdim,
                     Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  CHECK_STATUS(arg_create_output(input, dim, output));

  int64_t num_items = aitisa_tensor_size(input);
  int64_t input_ndim = aitisa_tensor_ndim(input);
  int64_t* input_dims = aitisa_tensor_dims(input);

  int64_t input_stride[input_ndim];
  aitisa_get_all_strides(input, input_stride);

  bool greater = false;
  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, arg_kernel, greater);
  if (!keepdim) {
    aitisa_squeeze(*output, NULL, 0, output);
  }
  return status;
  return status;
}