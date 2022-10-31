#include "src/new_ops1/image_transpose.h"
#include "src/core/dispatch.h"

#define transpose2d_kernel(typename)                   \
  typename* in_data = aitisa_tensor_data(input);       \
  typename* out_data = aitisa_tensor_data(*output);    \
  for (int64_t i = 0; i < n; i++) {                    \
    for (int64_t j = 0; j < m; j++) {                  \
      int64_t out_idx = i * m + j;                     \
      int64_t in_idx = j * n + i;                      \
      out_data[out_idx] = (typename)(in_data[in_idx]); \
    }                                                  \
  }

#define transpose3d_kernel(typename)                     \
  typename* in_data = aitisa_tensor_data(input);         \
  typename* out_data = aitisa_tensor_data(*output);      \
  for (int64_t c = 0; c < nchannel; c++) {               \
    int64_t channel_offset = m * n * c;                  \
    for (int64_t i = 0; i < n; i++) {                    \
      for (int64_t j = 0; j < m; j++) {                  \
        int64_t out_idx = channel_offset + i * m + j;    \
        int64_t in_idx = channel_offset + j * n + i;     \
        out_data[out_idx] = (typename)(in_data[in_idx]); \
      }                                                  \
    }                                                    \
  }

#define transpose4d_kernel(typename)                                       \
  typename* in_data = aitisa_tensor_data(input);                           \
  typename* out_data = aitisa_tensor_data(*output);                        \
  for (int64_t b = 0; b < batch_size; b++) {                               \
    for (int64_t c = 0; c < nchannel; c++) {                               \
      int64_t channel_offset = m * n * c;                                  \
      int64_t batch_offset = nchannel * m * n;                             \
      for (int64_t i = 0; i < n; i++) {                                    \
        for (int64_t j = 0; j < m; j++) {                                  \
          int64_t out_idx = b * batch_offset + channel_offset + i * m + j; \
          int64_t in_idx = b * batch_offset + channel_offset + j * n + i;  \
          out_data[out_idx] = (typename)(in_data[in_idx]);                 \
        }                                                                  \
      }                                                                    \
    }                                                                      \
  }

Status aitisa_image_transpose(const Tensor input, Tensor* output) {
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Status status = STATUS_SUCCESS;
  if (ndim == 2) {
    int m = dims[0];
    int n = dims[1];

    dims[0] = n;
    dims[1] = m;

    Tensor new_tensor;
    DataType dtype = aitisa_tensor_data_type(input);
    Device device = aitisa_tensor_device(input);

    CHECK_STATUS(
        aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
    *output = new_tensor;
    int64_t size = aitisa_tensor_size(input);
    AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, transpose2d_kernel);
  } else if (ndim == 3) {
    int nchannel = dims[0];
    int m = dims[1];
    int n = dims[2];

    dims[1] = n;
    dims[2] = m;

    Tensor new_tensor;
    DataType dtype = aitisa_tensor_data_type(input);
    Device device = aitisa_tensor_device(input);

    CHECK_STATUS(
        aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
    *output = new_tensor;
    int64_t size = aitisa_tensor_size(input);
    AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, transpose3d_kernel);
  } else if (ndim == 4) {
    int batch_size = dims[0];
    int nchannel = dims[1];
    int m = dims[2];
    int n = dims[3];

    dims[2] = n;
    dims[3] = m;

    Tensor new_tensor;
    DataType dtype = aitisa_tensor_data_type(input);
    Device device = aitisa_tensor_device(input);

    CHECK_STATUS(
        aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
    *output = new_tensor;
    int64_t size = aitisa_tensor_size(input);
    AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, transpose4d_kernel);
  } else {
    status = STATUS_TYPE_MISMATCH;
  }
  return status;
}
