#include "src/new_ops8/reduce_max.h"
#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/basic/index_utils.h"
#include "src/basic/squeeze.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/new_ops8/reduce_util.h"

static Status reduce_max_create_output(const Tensor input, const int64_t* dims,
                                       const int64_t dims_length,
                                       Tensor* output) {
  Status status;
  int64_t* input_dims = aitisa_tensor_dims(input);
  int64_t input_ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = kDouble;
  Device device = aitisa_tensor_device(input);
  int64_t mask[input_ndim];
  memset(mask, 1, sizeof(mask));
  for (int i = 0; i < dims_length; i++) {
    mask[dims[i]] = 0;
  }
  int64_t new_tensor_dims[input_ndim];
  memcpy(new_tensor_dims, input_dims, sizeof(int64_t) * input_ndim);
  for (int i = 0; i < dims_length; i++) {
    new_tensor_dims[dims[i]] = 1;
  }
  status = aitisa_full(dtype, device, new_tensor_dims, input_ndim, -999999,
                       &new_tensor);
  *output = new_tensor;
  return status;
}

#define reduce_max_kernel(typename, permuted_strides_in, permuted_strides_out, \
                          permuted_dims, num_items)                            \
  typename* in_data = (typename*)aitisa_tensor_data(input);                    \
  double* out_data = (double*)aitisa_tensor_data(*output);                     \
  if (input_ndim <= 1) {                                                       \
    for (int i = 0; i < input_dims[0]; i++) {                                  \
      typename* in_ptr = in_data + i;                                          \
      if (*out_data < *in_ptr)                                                 \
        *out_data = *in_ptr;                                                   \
    }                                                                          \
  } else {                                                                     \
    int count = 0;                                                             \
    int64_t step_value[input_ndim];                                            \
    memset(step_value, 0, sizeof(step_value));                                 \
    while (count < num_items) {                                                \
      typename* in_ptr_tmp = in_data;                                          \
      for (int i = 0; i < stride_length; i++) {                                \
        in_ptr_tmp += step_value[i] * permuted_strides_in[i];                  \
      }                                                                        \
      typename* in = in_ptr_tmp;                                               \
      double* out_ptr_tmp = out_data;                                          \
      for (int i = 0; i < stride_length; i++) {                                \
        out_ptr_tmp += step_value[i] * permuted_strides_out[i];                \
      }                                                                        \
      double* out = out_ptr_tmp;                                               \
      for (int i = 0; i < permuted_dims[1]; i++) {                             \
        for (int j = 0; j < permuted_dims[0]; j++) {                           \
          typename* in_ptr = in + j * permuted_strides_in[0];                  \
          double* out_ptr = out + j * permuted_strides_out[0];                 \
          if (*out_ptr < *in_ptr)                                              \
            *out_ptr = *in_ptr;                                                \
        }                                                                      \
        in = in + permuted_strides_in[1];                                      \
        out = out + permuted_strides_out[1];                                   \
      }                                                                        \
      count += permuted_dims[1] * permuted_dims[0];                            \
      for (int i = 2; i < input_ndim; i++) {                                   \
        if (step_value[i] < (permuted_dims[i] - 1)) {                          \
          step_value[i]++;                                                     \
          break;                                                               \
        }                                                                      \
        step_value[i] = 0;                                                     \
      }                                                                        \
    }                                                                          \
  }

Status aitisa_reduce_max(const Tensor input, const int64_t* dims,
                         const int64_t dims_length, const int keepdim,
                         Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  CHECK_STATUS(reduce_max_create_output(input, dims, dims_length, output));
  int64_t num_items = aitisa_tensor_size(input);
  int64_t input_ndim = aitisa_tensor_ndim(input);
  int64_t output_ndim = aitisa_tensor_ndim(*output);
  int64_t* input_dims = aitisa_tensor_dims(input);

  int64_t stride_length = int64_max(input_ndim, output_ndim);
  int64_t input_stride[input_ndim];
  int64_t output_stride[output_ndim];
  aitisa_get_all_strides(input, input_stride);
  aitisa_get_all_strides(*output, output_stride);

  int64_t input_stride_new[stride_length];
  int64_t output_stride_new[stride_length];
  memset(input_stride_new, 0, sizeof(input_stride_new));
  memset(output_stride_new, 0, sizeof(output_stride_new));

  strides_for_computing(input_stride, aitisa_tensor_dims(input), input_ndim,
                        stride_length, input_stride_new);
  strides_for_computing(output_stride, aitisa_tensor_dims(*output), output_ndim,
                        stride_length, output_stride_new);
  int64_t perm[stride_length];
  memset(perm, 0, sizeof(perm));

  reorder_dims(output_stride_new, stride_length, perm);

  int64_t permuted_strides_in[stride_length];
  int64_t permuted_strides_out[stride_length];
  int64_t permuted_dims[stride_length];
  permute_dims(input_stride_new, stride_length, perm, permuted_strides_in);
  permute_dims(output_stride_new, stride_length, perm, permuted_strides_out);
  permute_dims(input_dims, input_ndim, perm, permuted_dims);

  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, reduce_max_kernel,
                                   permuted_strides_in, permuted_strides_out,
                                   permuted_dims, num_items);
  if (!keepdim) {
    aitisa_squeeze(*output, NULL, 0, output);
  }
  return status;
}
