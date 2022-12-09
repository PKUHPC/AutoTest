#include "src/new_ops8/compare.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

void compare_int8_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(int8_t *)a) == (*(int8_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(int8_t *)a) > (*(int8_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(int8_t *)a) >= (*(int8_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(int8_t *)a) < (*(int8_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(int8_t *)a) <= (*(int8_t *)b);
      break;
    default:
      break;
  }
}

void compare_uint8_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(uint8_t *)a) == (*(uint8_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(uint8_t *)a) > (*(uint8_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(uint8_t *)a) >= (*(uint8_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(uint8_t *)a) < (*(uint8_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(uint8_t *)a) <= (*(uint8_t *)b);
      break;
    default:
      break;
  }
}

void compare_int16_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(int16_t *)a) == (*(int16_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(int16_t *)a) > (*(int16_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(int16_t *)a) >= (*(int16_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(int16_t *)a) < (*(int16_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(int16_t *)a) <= (*(int16_t *)b);
      break;
    default:
      break;
  }
}

void compare_uint16_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(uint16_t *)a) == (*(uint16_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(uint16_t *)a) > (*(uint16_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(uint16_t *)a) >= (*(uint16_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(uint16_t *)a) < (*(uint16_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(uint16_t *)a) <= (*(uint16_t *)b);
      break;
    default:
      break;
  }
}

void compare_int32_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(int32_t *)a) == (*(int32_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(int32_t *)a) > (*(int32_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(int32_t *)a) >= (*(int32_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(int32_t *)a) < (*(int32_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(int32_t *)a) <= (*(int32_t *)b);
      break;
    default:
      break;
  }
}

void compare_uint32_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(uint32_t *)a) == (*(uint32_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(uint32_t *)a) > (*(uint32_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(uint32_t *)a) >= (*(uint32_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(uint32_t *)a) < (*(uint32_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(uint32_t *)a) <= (*(uint32_t *)b);
      break;
    default:
      break;
  }
}

void compare_int64_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(int64_t *)a) == (*(int64_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(int64_t *)a) > (*(int64_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(int64_t *)a) >= (*(int64_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(int64_t *)a) < (*(int64_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(int64_t *)a) <= (*(int64_t *)b);
      break;
    default:
      break;
  }
}

void compare_uint64_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(uint64_t *)a) == (*(uint64_t *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(uint64_t *)a) > (*(uint64_t *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(uint64_t *)a) >= (*(uint64_t *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(uint64_t *)a) < (*(uint64_t *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(uint64_t *)a) <= (*(uint64_t *)b);
      break;
    default:
      break;
  }
}

void compare_float_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(float *)a) == (*(float *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(float *)a) > (*(float *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(float *)a) >= (*(float *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(float *)a) < (*(float *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(float *)a) <= (*(float *)b);
      break;
    default:
      break;
  }
}

void compare_double_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_EQUAL:
      *(int32_t *)c = (*(double *)a) == (*(double *)b);
      break;
    case OP_GREATER:
      *(int32_t *)c = (*(double *)a) > (*(double *)b);
      break;
    case OP_GREATER_EQUAL:
      *(int32_t *)c = (*(double *)a) >= (*(double *)b);
      break;
    case OP_LESS:
      *(int32_t *)c = (*(double *)a) < (*(double *)b);
      break;
    case OP_LESS_EQUAL:
      *(int32_t *)c = (*(double *)a) <= (*(double *)b);
      break;
    default:
      break;
  }
}

CompareOpFunc compare_op_func[TYPE_NTYPES] = {
    compare_int8_value,  compare_uint8_value,  compare_int16_value, compare_uint16_value,
    compare_int32_value, compare_uint32_value, compare_int64_value, compare_uint64_value,
    compare_float_value, compare_double_value};

CompareOpFunc aitisa_compare_op_func(DataType dtype) {
  return compare_op_func[dtype.code];
}

Status aitisa_equal(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(kInt32,
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  DataType output_dtype = aitisa_tensor_data_type(*output);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(output_dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_compare_op_func(dtype)(a, b, OP_EQUAL, c);
    aitisa_set_typed_array_value_func(output_dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_greater(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(kInt32,
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  DataType output_dtype = aitisa_tensor_data_type(*output);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(output_dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_compare_op_func(dtype)(a, b, OP_GREATER, c);
    aitisa_set_typed_array_value_func(output_dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_greater_equal(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(kInt32,
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  DataType output_dtype = aitisa_tensor_data_type(*output);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(output_dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_compare_op_func(dtype)(a, b, OP_GREATER_EQUAL, c);
    aitisa_set_typed_array_value_func(output_dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_less(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(kInt32,
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  DataType output_dtype = aitisa_tensor_data_type(*output);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(output_dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_compare_op_func(dtype)(a, b, OP_LESS, c);
    aitisa_set_typed_array_value_func(output_dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_less_equal(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(kInt32,
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  DataType output_dtype = aitisa_tensor_data_type(*output);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(output_dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_compare_op_func(dtype)(a, b, OP_LESS_EQUAL, c);
    aitisa_set_typed_array_value_func(output_dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}
