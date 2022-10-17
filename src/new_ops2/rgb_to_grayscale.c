#include "src/new_ops2/rgb_to_grayscale.h"
#include "src/core/dispatch.h"

#define rgb_to_grayscale3d_kernel(typename)                                                                                                             \
    typename *in_data = aitisa_tensor_data(input);                                                                                                             \
    typename *out_data = aitisa_tensor_data(*output);                                                                                                          \
    int64_t channel_offset = ncol * nrow;                                                                                                                      \
    for (int64_t i = 0; i < nrow; i++)                                                                                                                         \
    {                                                                                                                                                          \
        for (int64_t j = 0; j < ncol; j++)                                                                                                                     \
        {                                                                                                                                                      \
            int64_t idx = i * ncol + j;                                                                                                                        \
            out_data[idx] = 0.299 * in_data[channel_offset * 0 + idx] + 0.587 * in_data[channel_offset * 1 + idx] + 0.114 * in_data[channel_offset * 2 + idx]; \
        }                                                                                                                                                      \
    }

#define rgb_to_grayscale4d_kernel(typename)                                                                                                                              \
    typename *in_data = aitisa_tensor_data(input);                                                                                                                              \
    typename *out_data = aitisa_tensor_data(*output);                                                                                                                           \
    int64_t channel_offset = ncol * nrow;                                                                                                                                       \
    for (int64_t b = 0; b < batch_size; b++)                                                                                                                                    \
    {                                                                                                                                                                           \
        int64_t in_batch_offset = b * nchannel * nrow * ncol;                                                                                                                   \
        int64_t out_batch_offset = b * nrow * ncol;                                                                                                                             \
        for (int64_t i = 0; i < nrow; i++)                                                                                                                                      \
        {                                                                                                                                                                       \
            for (int64_t j = 0; j < ncol; j++)                                                                                                                                  \
            {                                                                                                                                                                   \
                int64_t in_idx = in_batch_offset + i * ncol + j;                                                                                                                \
                int64_t out_idx = out_batch_offset + i * ncol + j;                                                                                                              \
                out_data[out_idx] = 0.299 * in_data[channel_offset * 0 + in_idx] + 0.587 * in_data[channel_offset * 1 + in_idx] + 0.114 * in_data[channel_offset * 2 + in_idx]; \
            }                                                                                                                                                                   \
        }                                                                                                                                                                       \
    }

Status aitisa_rgb_to_grayscale(const Tensor input, Tensor *output)
{
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    if (ndim == 3) // [C, H, W]
    {
        int nchannel = dims[0];
        int nrow = dims[1];
        int ncol = dims[2];

        if (nchannel != 3)
        {
            status = STATUS_TYPE_MISMATCH;
            return status;
        }

        int64_t out_dims[2] = {nrow, ncol};

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, out_dims, ndim - 1, NULL, 0, &new_tensor));
        *output = new_tensor;
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rgb_to_grayscale3d_kernel);
    }
    else if (ndim == 4) // [N, C, H, W]
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

        int64_t out_dims[3] = {batch_size, nrow, ncol};

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, out_dims, ndim - 1, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, rgb_to_grayscale4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}