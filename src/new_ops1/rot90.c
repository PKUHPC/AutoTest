#include "src/new_ops1/rot90.h"
#include "src/core/dispatch.h"

#define rot90_3d_kernel0(typename)                    \
    typename *in_data = aitisa_tensor_data(input);    \
    typename *out_data = aitisa_tensor_data(*output); \
    for (int64_t i = 0; i < size; i++)                \
    {                                                 \
        out_data[i] = (typename)(in_data[i]);         \
    }

#define rot90_3d_kernel1(typename)                                                                     \
    typename *in_data = aitisa_tensor_data(input);                                                     \
    typename *out_data = aitisa_tensor_data(*output);                                                  \
    int64_t channel_offset = nrow * ncol;                                                              \
    for (int64_t i = 0; i < nrow; i++)                                                                 \
    {                                                                                                  \
        for (int64_t j = 0; j < ncol; j++)                                                             \
        {                                                                                              \
            int64_t in_idx = i * ncol + j;                                                             \
            int64_t out_idx = (ncol - 1 - j) * nrow + i;                                               \
            out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
            out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
            out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
        }                                                                                              \
    }

#define rot90_3d_kernel2(typename)                                                                     \
    typename *in_data = aitisa_tensor_data(input);                                                     \
    typename *out_data = aitisa_tensor_data(*output);                                                  \
    int64_t channel_offset = nrow * ncol;                                                              \
    for (int64_t i = 0; i < nrow; i++)                                                                 \
    {                                                                                                  \
        for (int64_t j = 0; j < ncol; j++)                                                             \
        {                                                                                              \
            int64_t in_idx = i * ncol + j;                                                             \
            int64_t out_idx = (nrow - 1 - i) * ncol + (ncol - 1 - j);                                  \
            out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
            out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
            out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
        }                                                                                              \
    }

#define rot90_3d_kernel3(typename)                                                                     \
    typename *in_data = aitisa_tensor_data(input);                                                     \
    typename *out_data = aitisa_tensor_data(*output);                                                  \
    int64_t channel_offset = nrow * ncol;                                                              \
    for (int64_t i = 0; i < nrow; i++)                                                                 \
    {                                                                                                  \
        for (int64_t j = 0; j < ncol; j++)                                                             \
        {                                                                                              \
            int64_t in_idx = i * ncol + j;                                                             \
            int64_t out_idx = j * nrow + (nrow - 1 - i);                                               \
            out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
            out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
            out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
        }                                                                                              \
    }


#define rot90_4d_kernel0(typename)                    \
    typename *in_data = aitisa_tensor_data(input);    \
    typename *out_data = aitisa_tensor_data(*output); \
    for (int64_t i = 0; i < size; i++)                \
    {                                                 \
        out_data[i] = (typename)(in_data[i]);         \
    }

#define rot90_4d_kernel1(typename)                                                                         \
    typename *in_data = aitisa_tensor_data(input);                                                         \
    typename *out_data = aitisa_tensor_data(*output);                                                      \
    int64_t channel_offset = nrow * ncol;                                                                  \
    for (int64_t b = 0; b < batch_size; b++)                                                               \
    {                                                                                                      \
        int64_t batch_offset = b * nchannel * ncol * nrow;                                                 \
        for (int64_t i = 0; i < nrow; i++)                                                                 \
        {                                                                                                  \
            for (int64_t j = 0; j < ncol; j++)                                                             \
            {                                                                                              \
                int64_t in_idx = batch_offset + i * ncol + j;                                              \
                int64_t out_idx = batch_offset + (ncol - 1 - j) * nrow + i;                                \
                out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
                out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
                out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
            }                                                                                              \
        }                                                                                                  \
    }

#define rot90_4d_kernel2(typename)                                                                         \
    typename *in_data = aitisa_tensor_data(input);                                                         \
    typename *out_data = aitisa_tensor_data(*output);                                                      \
    int64_t channel_offset = nrow * ncol;                                                                  \
    for (int64_t b = 0; b < batch_size; b++)                                                               \
    {                                                                                                      \
        int64_t batch_offset = b * nchannel * ncol * nrow;                                                 \
        for (int64_t i = 0; i < nrow; i++)                                                                 \
        {                                                                                                  \
            for (int64_t j = 0; j < ncol; j++)                                                             \
            {                                                                                              \
                int64_t in_idx = batch_offset + i * ncol + j;                                              \
                int64_t out_idx = batch_offset + (nrow - 1 - i) * ncol + (ncol - 1 - j);                   \
                out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
                out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
                out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
            }                                                                                              \
        }                                                                                                  \
    }

#define rot90_4d_kernel3(typename)                                                                         \
    typename *in_data = aitisa_tensor_data(input);                                                         \
    typename *out_data = aitisa_tensor_data(*output);                                                      \
    int64_t channel_offset = nrow * ncol;                                                                  \
    for (int64_t b = 0; b < batch_size; b++)                                                               \
    {                                                                                                      \
        int64_t batch_offset = b * nchannel * ncol * nrow;                                                 \
        for (int64_t i = 0; i < nrow; i++)                                                                 \
        {                                                                                                  \
            for (int64_t j = 0; j < ncol; j++)                                                             \
            {                                                                                              \
                int64_t in_idx = batch_offset + i * ncol + j;                                              \
                int64_t out_idx = batch_offset + j * nrow + (nrow - 1 - i);                                \
                out_data[out_idx] = (typename)(in_data[in_idx]);                                           \
                out_data[out_idx + channel_offset] = (typename)(in_data[in_idx + channel_offset]);         \
                out_data[out_idx + 2 * channel_offset] = (typename)(in_data[in_idx + 2 * channel_offset]); \
            }                                                                                              \
        }                                                                                                  \
    }

Status aitisa_rot90(const Tensor input, const int k, Tensor *output)
{
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Status status = STATUS_SUCCESS;
  if(ndim==3)
  {
    int nchannel = dims[0];
    int nrow = dims[1];
    int ncol = dims[2];

    if (nchannel != 3)
    {
      status = STATUS_TYPE_MISMATCH;
      return status;
    }

    int64_t *new_dims[3] = {nchannel, ncol, nrow};

    Tensor new_tensor;
    DataType dtype = aitisa_tensor_data_type(input);
    Device device = aitisa_tensor_device(input);

    if(k%4==0)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_3d_kernel0);
    }
    else if(k%4==1)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_3d_kernel1);
    }
    else if(k%4==2)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_3d_kernel2);
    }
    else
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_3d_kernel3);
    }

  }
  else if(ndim==4)
  {
    int batch_size = dims[0];
    int nchannel = dims[1];
    int nrow = dims[2];
    int ncol = dims[3];

    if (nchannel != 3)
    {
      status = STATUS_TYPE_MISMATCH;
      return status;
    }

    int64_t *new_dims[4] = {batch_size, nchannel, ncol, nrow};

    Tensor new_tensor;
    DataType dtype = aitisa_tensor_data_type(input);
    Device device = aitisa_tensor_device(input);

    if(k%4==0)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_4d_kernel0);
    }
    else if(k%4==1)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_4d_kernel1);
    }
    else if(k%4==2)
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_4d_kernel2);
    }
    else
    {
      CHECK_STATUS(
          aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
      *output = new_tensor;
      int64_t size = aitisa_tensor_size(input);
      AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rot90_4d_kernel3);
    }

  }
  else
  {
    status = STATUS_TYPE_MISMATCH;
  }
  return status;
}