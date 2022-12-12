#include "src/new_ops8/reduce_sum.h"
#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/basic/index_utils.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status permute_dims(const int64_t* stride, const int64_t stride_length,
                           const int64_t* perm, int64_t* permuted_strides) {
  for (int i = 0; i < stride_length; i++) {
    permuted_strides[i] = stride[perm[i]];
  }
}

static Status should_swap(const int64_t* strides, int64_t dim0, int64_t dim1) {
  printf("strides[dim0] = %ld  || strides[dim0] = %ld \n ", strides[dim0],
         strides[dim1]);

  return strides[dim0] < strides[dim1];
}
static Status int64_swap(int64_t* a, int64_t* b) {
  int64_t tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
}
static Status reorder_dims(int64_t* strides, int64_t strides_length,
                           int64_t* perm) {
  for (int i = 0; i < strides_length; i++) {
    perm[i] = i;
  }
  for (int i = 1; i < strides_length; ++i) {
    for (int j = i; j > 0; j--) {
      printf("j == %d", j);
      printf("perm[j] = %ld || perm[j-1] = %ld \n", perm[j], perm[j - 1]);
      int comparsion = should_swap(strides, perm[j], perm[j - 1]);
      if (comparsion) {
        printf("yes\n");
        int64_swap(perm + j, perm + j - 1);
      } else {
        break;
      }
    }
  }
}

static Status strides_for_computing(const int64_t* strides_old,
                                    const int64_t* dims, const int64_t ndim,
                                    int64_t length_stride,
                                    int64_t* strides_new) {
  int64_t offset = length_stride - ndim;
  memset(strides_new, 0, sizeof(strides_new));
  for (int j = 0; j < ndim; ++j) {
    if (dims[j] != 1) {
      strides_new[j + offset] = strides_old[j];
    }
  }
}

int64_t int64_max(int64_t x, int64_t y) {
  if (x > y)
    return x;
  else
    return y;
}

static Status reduce_sum_create_output(const Tensor input, const int64_t* dims,
                                       const int64_t dims_length,
                                       const int keepdim, Tensor* output) {
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
  if (keepdim) {
    int64_t new_tensor_dims[input_ndim];
    memcpy(new_tensor_dims, input_dims, sizeof(int64_t) * input_ndim);
    for (int i = 0; i < dims_length; i++) {
      new_tensor_dims[dims[i]] = 1;
    }
    status =
        aitisa_full(dtype, device, new_tensor_dims, input_ndim, 0, &new_tensor);
  } else {
    int64_t new_tensor_dims[input_ndim - dims_length];
    int64_t* ptr = new_tensor_dims;
    for (int i = 0; i < input_ndim; i++) {
      if (mask[i]) {
        *ptr = input_dims[i];
        ptr++;
      }
    }
    status = aitisa_full(dtype, device, new_tensor_dims,
                         input_ndim - dims_length, 0, &new_tensor);
  }
  *output = new_tensor;
  return status;
}

#define reduce_sum_kernel(typename, permuted_strides_in, permuted_strides_out, \
                          permuted_dims, num_items)                            \
  typename* in_data = (typename*)aitisa_tensor_data(input);                    \
  double* out_data = (double*)aitisa_tensor_data(*output);                     \
  if (input_ndim <= 1) {                                                       \
    for (int i = 0; i < input_dims[0]; i++) {                                  \
      typename* in_ptr = in_data + i;                                          \
      *out_data = *out_data + *in_ptr;                                         \
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
          *out_ptr = *out_ptr + *in_ptr;                                       \
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

Status aitisa_reduce_sum(const Tensor input, const int64_t* dims,
                         const int64_t dims_length, const int keepdim,
                         Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  CHECK_STATUS(
      reduce_sum_create_output(input, dims, dims_length, keepdim, output));
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

  for (int64_t i = 0; i < stride_length; i++) {
    printf("#%ld, ", output_stride_new[i]);
  }
  printf("\n");
  for (int64_t i = 0; i < stride_length; i++) {
    printf("$%ld, ", input_stride[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < stride_length; i++) {
    printf("$%ld, ", output_stride[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < stride_length; i++) {
    printf("(%ld, ", input_stride_new[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < stride_length; i++) {
    printf("(%ld,", output_stride_new[i]);
  }
  printf("\n");

  for (int64_t i = 0; i < stride_length; i++) {
    printf("&%ld,", perm[i]);
  }
  printf("\n");

  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, reduce_sum_kernel,
                                   permuted_strides_in, permuted_strides_out,
                                   permuted_dims, num_items);

  return status;
}
