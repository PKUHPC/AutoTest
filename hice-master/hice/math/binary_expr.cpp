#include "hice/math/binary_expr.h"

namespace hice {

HICE_DEFINE_DISPATCHER(add_dispatcher);
HICE_DEFINE_DISPATCHER(sub_dispatcher);
HICE_DEFINE_DISPATCHER(mul_dispatcher);
HICE_DEFINE_DISPATCHER(div_dispatcher);
HICE_DEFINE_DISPATCHER(max_dispatcher);

/* ====outplace==== */
// tensor, tensor
Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
  // std::cout << "In add" << std::endl;
  Tensor result(device(tensor1.device()).dtype(tensor1.data_type()));
  add_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor sub(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(tensor1.data_type()));
  sub_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor mul(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(tensor1.data_type()));
  mul_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor div(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(tensor1.data_type()));
  div_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

Tensor max(const Tensor& tensor1, const Tensor& tensor2) {
  Tensor result(device(tensor1.device()).dtype(tensor1.data_type()));
  max_dispatcher(tensor1, tensor2, result, /* resizable = */true);
  return result;
}

// tensor, scalar
Tensor add(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return add(tensor1, tensor2);
}

Tensor sub(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return sub(tensor1, tensor2);
}

Tensor mul(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return mul(tensor1, tensor2);
}

Tensor div(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return div(tensor1, tensor2);
}

Tensor max(const Tensor& tensor1, Scalar scalar) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return max(tensor1, tensor2);
}

// scalar, tensor
Tensor add(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return add(tensor1, tensor2);
}

Tensor sub(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return sub(tensor1, tensor2);
}

Tensor mul(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return mul(tensor1, tensor2);
}

Tensor div(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return div(tensor1, tensor2);
}

Tensor max(Scalar scalar, const Tensor& tensor2) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return max(tensor1, tensor2);
}

/* ====inplace==== */
// tensor, tensor, result
Tensor& add(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  add_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& sub(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  sub_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& mul(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  mul_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& div(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  div_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

Tensor& max(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  max_dispatcher(tensor1, tensor2, result, /* resizable = */false);
  return result;
}

// tensor, scalar, result
Tensor& add(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return add(tensor1, tensor2, result);
}

Tensor& sub(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return sub(tensor1, tensor2, result);
}

Tensor& mul(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return mul(tensor1, tensor2, result);
}

Tensor& div(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return div(tensor1, tensor2, result);
}

Tensor& max(const Tensor& tensor1, Scalar scalar, Tensor& result) {
  Tensor tensor2 = scalar_to_tensor(scalar, 
                                    tensor1.scalar_type(),
                                    tensor1.device_type());
  return max(tensor1, tensor2, result);
}

// scalar, tensor, result
Tensor& add(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return add(tensor1, tensor2, result);
}

Tensor& sub(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return sub(tensor1, tensor2, result);
}

Tensor& mul(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return mul(tensor1, tensor2, result);
}

Tensor& div(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return div(tensor1, tensor2, result);
}

Tensor& max(Scalar scalar, const Tensor& tensor2, Tensor& result) {
  Tensor tensor1 = scalar_to_tensor(scalar, 
                                    tensor2.scalar_type(),
                                    tensor2.device_type());
  return max(tensor1, tensor2, result);
}


} // namespace hice
