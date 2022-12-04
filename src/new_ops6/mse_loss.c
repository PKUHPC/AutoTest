#include "src/new_ops6/mse_loss.h"
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/math/binary_op.h"
#include "src/new_ops7/square.h"

static Status mse_loss_create_output(const Tensor input, Tensor* output) {
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

Status aitisa_mse_loss(const Tensor input, const Tensor target,
                      const Tensor weight, Tensor* output){

  Status status = STATUS_SUCCESS;
  mse_loss_create_output(input,output);
  aitisa_sub(input, target,output);
  aitisa_square(*output, output);
  if(weight){
    aitisa_mul(*output,weight,output);
  }

  return status;
}