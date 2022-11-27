#include "src/new_ops6/cross_entropy.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status cross_entropy_create_output(const Tensor input, Tensor* output) {
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  int64_t new_tensor_dims[1] = {dims[0]};
  status = aitisa_full(dtype, device, new_tensor_dims, 1, 0, &new_tensor);
  *output = new_tensor;

  return status;
}

#define cross_entropy_kernel(typename)                                     \
  typename* weight_data = NULL;                                            \
  if (weight) {                                                            \
    weight_data = aitisa_tensor_data(weight);                              \
  }                                                                        \
  float* target_data = aitisa_tensor_data(target);                         \
  typename* in_data = aitisa_tensor_data(prob);                            \
  typename* out_data = aitisa_tensor_data(*output);                        \
  for (int64_t index = 0; index < batch_size * class_size; ++index) {      \
    int64_t inner_index = index % class_size;                              \
    int64_t outer_index = index / class_size;                              \
    float weight = (weight_data == NULL ? 1.0 : weight_data[inner_index]); \
    out_data[outer_index] +=                                               \
        -log(in_data[index]) * target_data[index] * weight;                \
  }

Status aitisa_cross_entropy(const Tensor prob, const Tensor target,
                            const Tensor weight, Tensor* output) {

  Status status = STATUS_SUCCESS;
  DataType dtype = aitisa_tensor_data_type(prob);
  int64_t batch_size = aitisa_tensor_dims(prob)[0];
  int64_t class_size = aitisa_tensor_dims(prob)[1];
  CHECK_STATUS(cross_entropy_create_output(prob, output));
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, cross_entropy_kernel);
  return status;
}