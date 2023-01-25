#include "src/new_ops6/smooth_l1_loss.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/math/binary_op.h"
#include "src/new_ops7/abs.h"
#include "src/new_ops8/reduce_mean.h"
#include "src/new_ops8/reduce_sum.h"

static Status smooth_l1_loss_create_output(const Tensor input, Tensor* output) {
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

#define smooth_l1_loss_kernel(typename)                                    \
  typename* in_data = aitisa_tensor_data(tmp);                             \
  typename* out_data = aitisa_tensor_data(*output);                        \
  int64_t size = aitisa_tensor_size(input);                                \
  for (int64_t i = 0; i < size; i++) {                                     \
    out_data[i] =                                                          \
        in_data[i] < 1 ? 0.5 * in_data[i] * in_data[i] : in_data[i] - 0.5; \
  }

Status aitisa_smooth_l1_loss(const Tensor input, const Tensor target,
                             const Tensor weight, const int reduction,
                             Tensor* output) {

  Status status = STATUS_SUCCESS;
  Tensor tmp;
  smooth_l1_loss_create_output(input, &tmp);
  aitisa_sub(input, target, &tmp);
  aitisa_abs(tmp, &tmp);
  smooth_l1_loss_create_output(tmp, output);
  DataType dtype = aitisa_tensor_data_type(tmp);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, smooth_l1_loss_kernel);
  aitisa_destroy(&tmp);
  if (weight) {
    aitisa_mul(*output, weight, output);
  }
  int64_t reduce_ndim = aitisa_tensor_ndim(*output);
  int64_t reduce_dims[reduce_ndim];
  for (int64_t i = 0; i < reduce_ndim; i++) {
    reduce_dims[i] = i;
  }
  if (reduction == 1) {
    aitisa_reduce_mean(*output, reduce_dims, reduce_ndim, 0, output);
  } else if (reduction == 2) {
    aitisa_reduce_sum(*output, reduce_dims, reduce_ndim, 0, output);
  }
  return status;
}