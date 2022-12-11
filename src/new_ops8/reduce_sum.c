#include "src/new_ops8/reduce_sum.h"
#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/basic/index_utils.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
static Status reorder_dims(int64_t* strides) {
//  int ndim = strides.size();
//  std::vector<int64_t> perm_(ndim, 0);
//  // ndim-1, ndim-2, ..., 0
//  std::iota(perm_.rbegin(), perm_.rend(), 0);
//  auto should_swap = [&](int64_t dim0, int64_t dim1) {
//    return strides[dim0] < strides[dim1];
//  };
//  for (int i = 1; i < ndim; i++) {
//    for (int j = i; j > 0; j--) {
//      bool comparison = should_swap(perm_[j], perm_[j - 1]);
//      if (comparison) {
//        std::swap(perm_[j], perm_[j - 1]);
//      } else {
//        break;
//      }
//    }
//  }
//  return perm_;
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
  DataType dtype = aitisa_tensor_data_type(input);
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

#define reduce_sum_kernel(typename)

Status aitisa_reduce_sum(const Tensor input, const int64_t* dims,
                         const int64_t dims_length, const int keepdim,
                         Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  CHECK_STATUS(
      reduce_sum_create_output(input, dims, dims_length, keepdim, output));
  int64_t input_ndim = aitisa_tensor_ndim(input);
  int64_t output_ndim = aitisa_tensor_ndim(*output);

  int64_t length_stride = int64_max(input_ndim, output_ndim);
  int64_t input_stride[input_ndim];
  int64_t output_stride[output_ndim];
  aitisa_get_all_strides(input, input_stride);
  aitisa_get_all_strides(*output, output_stride);

  int64_t input_stride_new[length_stride];
  int64_t output_stride_new[length_stride];
  memset(input_stride_new, 0, sizeof(input_stride_new));
  memset(output_stride_new, 0, sizeof(output_stride_new));

  strides_for_computing(input_stride, aitisa_tensor_dims(input), input_ndim,
                        length_stride, input_stride_new);
  strides_for_computing(output_stride, aitisa_tensor_dims(*output), output_ndim,
                        length_stride, output_stride_new);

//  for (int64_t i = 0; i < length_stride; i++) {
//    printf("%ld, ", input_stride_new[i]);
//  }
//
//  for (int64_t i = 0; i < length_stride; i++) {
//    printf("%ld,", output_stride_new[i]);
//  }
  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, reduce_sum_kernel);

  return status;
}
