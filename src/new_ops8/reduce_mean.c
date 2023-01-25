#include "src/new_ops8/reduce_mean.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/new_ops8/reduce_sum.h"

#define reduce_mean_kernel(typename, dims, dims_length, num_items) \
  typename* in_data = (typename*)aitisa_tensor_data(input);        \
  typename* out_data = (typename*)aitisa_tensor_data(*output);         \
  int64_t* input_dims = aitisa_tensor_dims(input);                 \
  typename factor = 1;                                               \
  for (int64_t i = 0; i < dims_length; i++) {                      \
    factor *= input_dims[dims[i]];                                 \
  }                                                                \
                                                                   \
  for (int i = 0; i < num_items; i++) {                            \
    out_data[i] /= factor;                                         \
  }

Status aitisa_reduce_mean(const Tensor input, const int64_t* dims,
                          const int64_t dims_length, const int keepdim,
                          Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(input);
  status = aitisa_reduce_sum(input, dims, dims_length, keepdim, output);
  int64_t num_items = aitisa_tensor_size(*output);
  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, reduce_mean_kernel, dims,
                                   dims_length, num_items);

  return status;
}
