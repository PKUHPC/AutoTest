#include "hice/math/compare.h"

namespace hice {

HICE_DEFINE_DISPATCHER(equal_dispatcher);
HICE_DEFINE_DISPATCHER(less_dispatcher);
HICE_DEFINE_DISPATCHER(less_equal_dispatcher);
HICE_DEFINE_DISPATCHER(greater_dispatcher);
HICE_DEFINE_DISPATCHER(greater_equal_dispatcher);

/* ====outplace==== */
// tensor, tensor
Tensor equal(const Tensor& tensor1, const Tensor& tensor2) {
  // std::cout << "In equal" << std::endl;
  Tensor result(device(tensor1.device()).dtype(kBool));
  equal_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor less(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(kBool));
  less_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor less_equal(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(kBool));
  less_equal_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor greater(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(kBool));
  greater_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor greater_equal(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(kBool));
  greater_equal_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

// tensor, scalar
Tensor equal(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return equal(tensor1, tensor2);
}

Tensor less(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return less(tensor1, tensor2);
}

Tensor less_equal(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return less_equal(tensor1, tensor2);
}

Tensor greater(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return greater(tensor1, tensor2);
}

Tensor greater_equal(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return greater_equal(tensor1, tensor2);
}

// scalar, tensor
Tensor equal(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return equal(tensor1, tensor2);
}

Tensor less(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return less(tensor1, tensor2);
}

Tensor less_equal(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return less_equal(tensor1, tensor2);
}

Tensor greater(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return greater(tensor1, tensor2);
}

Tensor greater_equal(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return greater_equal(tensor1, tensor2);
}

/* ====inplace==== */
// tensor, tensor, result
Tensor& equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  equal_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& less(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  less_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& less_equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  less_equal_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& greater(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  greater_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& greater_equal(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  greater_equal_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

// tensor, scalar, result
Tensor& equal(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return equal(tensor1, tensor2, result);
}

Tensor& less(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return less(tensor1, tensor2, result);
}

Tensor& less_equal(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return less_equal(tensor1, tensor2, result);
}

Tensor& greater(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return greater(tensor1, tensor2, result);
}

Tensor& greater_equal(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return greater_equal(tensor1, tensor2, result);
}

// scalar, tensor, result
Tensor& equal(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return equal(tensor1, tensor2, result);
}

Tensor& less(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return less(tensor1, tensor2, result);
}

Tensor& less_equal(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return less_equal(tensor1, tensor2, result);
}

Tensor& greater(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return greater(tensor1, tensor2, result);
}

Tensor& greater_equal(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return greater_equal(tensor1, tensor2, result);
}


} // namespace hice
