#include "hice/api_c/ops_core.h"
#include "hice/api_c/tensor_impl.h"
#include "hice/api_c/error_handle.h"

#include "hice/core/tensor_printer.h"

HI_Status HI_TensorSize(const HI_Tensor tensor1, int64_t *size) {
  HI_API_BEGIN();
  *size = tensor1->tensor_.size();
  HI_API_END();
}

HI_Status HI_TensorItemSize(const HI_Tensor tensor1, size_t *item_size) {
  HI_API_BEGIN();
  *item_size = tensor1->tensor_.item_size();
  HI_API_END();
}

HI_Status HI_TensorDims(const HI_Tensor tensor1, const int64_t **dims) {
  HI_API_BEGIN();
  *dims = tensor1->tensor_.dims().data();
  HI_API_END();
}

HI_Status HI_TensorNdim(const HI_Tensor tensor1, int64_t *ndim) {
  HI_API_BEGIN();
  *ndim = tensor1->tensor_.ndim();
  HI_API_END();
}

HI_Status HI_TensorRawMutableData(HI_Tensor tensor1, void **raw_data) {
  HI_API_BEGIN();
  (*raw_data) = tensor1->tensor_.raw_mutable_data();
  HI_API_END();
}

HI_Status HI_Print(const HI_Tensor tensor1) {
  HI_API_BEGIN();
  hice::TensorPrinter tp;
  tp.print(tensor1->tensor_);
  HI_API_END();
}

HI_Status HI_ToDevice(HI_Tensor input, HI_Device hi_device, HI_Tensor *output) {
  HI_API_BEGIN();
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  *output = new HI_Tensor_Impl{input->tensor_.to(device_type)};
  HI_API_END();
}