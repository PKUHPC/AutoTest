#include <hice/util/dlpack_wrapper.h>

namespace hice {

DLTensor HICETensor_to_DLTensor(const hice::Tensor& self_const) {
  hice::Tensor& self = const_cast<hice::Tensor&>(self_const);
  DLDeviceType device_type;
  switch(self.device_type()) {
    case kCPU: 
      device_type = kDLCPU;
      break;
    case kCUDA: 
      device_type = kDLGPU;
      break;
  }

  DLDataTypeCode data_code;
  auto sc_type = self.scalar_type();
  if (sc_type == kInt8 || sc_type == kInt16 || sc_type == kInt32) {
    data_code = kDLInt;
  } else if (sc_type == kUInt8 || sc_type == kUInt16 || sc_type == kUInt32) {
    data_code = kDLUInt;
  } else {
    HICE_CHECK_SUPPORTED(sc_type == kFloat) << "UnSupported type when convert hice tensor to dltensor.";
    data_code = kDLFloat;
  }
  
  DLTensor dltensor;
  dltensor.data = self.raw_mutable_data();  
  int device_id = self.device().index() == -1 ? 0 : self.device().index();
  dltensor.ctx = {device_type, device_id};
  dltensor.ndim = self.ndim();  
  dltensor.dtype = {data_code, static_cast<uint8_t>(self.item_size() * 8), 1};
  dltensor.shape = self.mutable_impl().mutable_shape().mutable_dimensions().data();
  dltensor.strides = nullptr;  
  dltensor.byte_offset = 0;  

  return dltensor;
}

DLManagedTensor HICETensor_to_DLManagedTensor(const hice::Tensor& self_const) {
  DLTensor dl_tensor = HICETensor_to_DLTensor(self_const);
  DLManagedTensor dlm_tensor = {dl_tensor, nullptr, nullptr};
  return dlm_tensor;
}

// NOTE: strides and byte_offset are ignored
Tensor DLTensor_to_HICETensor(const DLTensor& self_const) {
  DLTensor& self = const_cast<DLTensor&>(self_const);
  void* data_ptr = self.data;
  DLContext& dl_ctx = self.ctx;
  DLDataType& dl_date_type = self.dtype;
  int ndim = self.ndim;
  std::vector<int64_t> dims(self.shape, self.shape + ndim);

  uint8_t device_id = dl_ctx.device_id;
  DeviceType device_type;
  switch(dl_ctx.device_type) {
    case kDLCPU: 
      device_type = kCPU;
      break;
    case kDLGPU: 
      device_type = kCUDA;
      break;
  }

  HICE_CHECK_EQ(dl_date_type.lanes, 1) << "Not supported DLDataType";
  ScalarType sc_type;
  if (dl_date_type.code == kDLInt && dl_date_type.bits == 8) {
    sc_type = kInt8;
  } else if (dl_date_type.code == kDLInt && dl_date_type.bits == 16) {
    sc_type = kInt16;
  } else if (dl_date_type.code == kDLInt && dl_date_type.bits == 32) {
    sc_type = kInt32;
  } else if (dl_date_type.code == kDLUInt && dl_date_type.bits == 8) {
    sc_type = kUInt8;
  } else if (dl_date_type.code == kDLUInt && dl_date_type.bits == 16) {
    sc_type = kUInt16;
  } else if (dl_date_type.code == kDLUInt && dl_date_type.bits == 32) {
    sc_type = kUInt32;
  } else if (dl_date_type.code == kDLFloat) {
    sc_type = kFloat;
  } else {
    HICE_CHECK_SUPPORTED(false);
  }

  TensorOptions options = device({device_type, device_id}).dtype(sc_type);
  Tensor tensor = wrap(dims, data_ptr, options, false);
  return tensor;
}

} // namespace hice