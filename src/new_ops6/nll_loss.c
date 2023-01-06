#include "src/new_ops6/nll_loss.h"
#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status nll_loss_create_output(const Tensor input, Tensor* output) {
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  if (ndim == 2) {
    int64_t new_tensor_dims[1] = {dims[0]};
    status = aitisa_full(dtype, device, new_tensor_dims, 1, 0, &new_tensor);
    *output = new_tensor;
  } else if (ndim == 4) {
    int64_t new_tensor_dims[3] = {dims[0], dims[2], dims[3]};
    status = aitisa_full(dtype, device, new_tensor_dims, 3, 0, &new_tensor);
    *output = new_tensor;
  }
  return status;
}

#define nll_cross_kernel(typename)                               \
  typename* probs_data = aitisa_tensor_data(probs);              \
  int64_t* target_data = (int64_t*)aitisa_tensor_data(target);   \
  typename* loss_data = aitisa_tensor_data(*loss);               \
  typename* weight_data = NULL;                                  \
  if (weight) {                                                  \
    weight_data = aitisa_tensor_data(weight);                    \
  }                                                              \
  int64_t loss_size = aitisa_tensor_size(*loss);                 \
  int64_t ndim = aitisa_tensor_ndim(probs);                      \
  int64_t n_class = aitisa_tensor_dims(probs)[ndim - 1];         \
  for (int64_t idx_loss = 0; idx_loss < loss_size; idx_loss++) { \
    int64_t trg = target_data[idx_loss];                         \
    int64_t offset_ipt = idx_loss * n_class + trg;               \
    typename ipt = probs_data[offset_ipt];                       \
    typename wgt = weight_data == NULL ? 1 : weight_data[trg];   \
    loss_data[idx_loss] = -1 * wgt * ipt;                        \
  }

Status aitisa_nll_loss(const Tensor probs, const Tensor target,
                       const Tensor weight, Tensor* loss) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(probs);
  CHECK_STATUS(nll_loss_create_output(probs, loss));
  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, nll_cross_kernel);

  return status;
}
