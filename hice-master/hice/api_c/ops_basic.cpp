#include "hice/api_c/ops_basic.h"
#include "hice/api_c/error_handle.h"
#include "hice/api_c/tensor_impl.h"
#include "hice/basic/factories.h"

HI_Status HI_Create(HI_DataType hi_data_type, HI_Device hi_device,
                    int64_t *dims, int64_t ndim, void *data, size_t len,
                    HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ScalarType scalar_type =
      static_cast<hice::ScalarType>(hi_data_type.code);
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  std::vector<int64_t> dims_vec(dims, dims + ndim);
  *output = new HI_Tensor_Impl{hice::create(
      dims_vec, data, len, dtype(scalar_type).device(device_type))};
  HI_API_END();
}

HI_Status HI_Wrap(HI_DataType hi_data_type, HI_Device hi_device, int64_t *dims,
                  int64_t ndim, void *data, HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ScalarType scalar_type =
      static_cast<hice::ScalarType>(hi_data_type.code);
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  std::vector<int64_t> dims_vec(dims, dims + ndim);
  *output = new HI_Tensor_Impl{
      hice::wrap(dims_vec, data, dtype(scalar_type).device(device_type))};
  HI_API_END();
}

HI_Status HI_Full(HI_DataType hi_data_type, HI_Device hi_device, int64_t *dims,
                  int64_t ndim, void *value, HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ScalarType scalar_type =
      static_cast<hice::ScalarType>(hi_data_type.code);
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  std::vector<int64_t> dims_vec(dims, dims + ndim);
  HICE_DISPATCH_ALL_TYPES(scalar_type, "HI_FULL", [&]() {
    scalar_t *val_ptr = static_cast<scalar_t *>(value);
    *output = new HI_Tensor_Impl{
        hice::full(dims_vec, *val_ptr, dtype(scalar_type).device(device_type))};
  });
  HI_API_END();
}

HI_Status HI_RandUniform(HI_DataType hi_data_type, HI_Device hi_device,
                         int64_t *dims, int64_t ndim, void *a, void *b,
                         HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ScalarType scalar_type =
      static_cast<hice::ScalarType>(hi_data_type.code);
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  std::vector<int64_t> dims_vec(dims, dims + ndim);
  HICE_DISPATCH_ALL_TYPES(scalar_type, "HI_RANDUniform", [&]() {
    scalar_t *a_ptr = static_cast<scalar_t *>(a);
    scalar_t *b_ptr = static_cast<scalar_t *>(b);
    *output = new HI_Tensor_Impl{hice::rand_uniform(
        dims_vec, *a_ptr, *b_ptr, dtype(scalar_type).device(device_type))};
  });
  HI_API_END();
}

HI_Status HI_RandNormal(HI_DataType hi_data_type, HI_Device hi_device,
                        int64_t *dims, int64_t ndim, void *mean, void *stddev,
                        HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ScalarType scalar_type =
      static_cast<hice::ScalarType>(hi_data_type.code);
  hice::DeviceType device_type = static_cast<hice::DeviceType>(hi_device.type);
  std::vector<int64_t> dims_vec(dims, dims + ndim);
  HICE_DISPATCH_ALL_TYPES(scalar_type, "HI_RandNormal", [&]() {
    scalar_t *mean_ptr = static_cast<scalar_t *>(mean);
    scalar_t *stddev_ptr = static_cast<scalar_t *>(stddev);
    *output = new HI_Tensor_Impl{
        hice::rand_normal(dims_vec, *mean_ptr, *stddev_ptr,
                          dtype(scalar_type).device(device_type))};
  });
  HI_API_END();
}