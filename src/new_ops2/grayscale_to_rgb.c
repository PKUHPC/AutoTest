#include "src/new_ops2/grayscale_to_rgb.h"
#include "src/core/dispatch.h"

#define grayscale_to_rgb3d_kernel(typename)             \
    typename *in_data = aitisa_tensor_data(input);             \
    typename *out_data = aitisa_tensor_data(*output);          \
    int64_t channel_offset = ncol * nrow;                      \
    for (int64_t i = 0; i < nrow; i++)                         \
    {                                                          \
        for (int64_t j = 0; j < ncol; j++)                     \
        {                                                      \
            int64_t idx = i * ncol + j;                        \
            out_data[channel_offset * 0 + idx] = in_data[idx]; \
            out_data[channel_offset * 1 + idx] = in_data[idx]; \
            out_data[channel_offset * 2 + idx] = in_data[idx]; \
        }                                                      \
    }

#define grayscale_to_rgb4d_kernel(typename)                        \
    typename *in_data = aitisa_tensor_data(input);                        \
    typename *out_data = aitisa_tensor_data(*output);                     \
    int64_t channel_offset = ncol * nrow;                                 \
    for (int64_t b = 0; b < batch_size; b++)                              \
    {                                                                     \
        int64_t in_batch_offset = b * nrow * ncol;                        \
        int64_t out_batch_offset = b * nchannel * nrow * ncol;            \
        for (int64_t i = 0; i < nrow; i++)                                \
        {                                                                 \
            for (int64_t j = 0; j < ncol; j++)                            \
            {                                                             \
                int64_t in_idx = in_batch_offset + i * ncol + j;          \
                int64_t out_idx = out_batch_offset + i * ncol + j;        \
                out_data[channel_offset * 0 + out_idx] = in_data[in_idx]; \
                out_data[channel_offset * 1 + out_idx] = in_data[in_idx]; \
                out_data[channel_offset * 2 + out_idx] = in_data[in_idx]; \
            }                                                             \
        }                                                                 \
    }

Status aitisa_grayscale_to_rgb(const Tensor input, Tensor *output)
{
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    if (ndim == 2) // [H, W]
    {
        int nchannel = 3;
        int nrow = dims[0];
        int ncol = dims[1];

        int64_t out_dims[3] = {nchannel, nrow, ncol};

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, out_dims, ndim + 1, NULL, 0, &new_tensor));
        *output = new_tensor;
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, grayscale_to_rgb3d_kernel);
    }
    else if (ndim == 3) // [N, H, W]
    {
        int batch_size = dims[0];
        int nchannel = 3;
        int nrow = dims[1];
        int ncol = dims[2];

        int64_t out_dims[4] = {batch_size, nchannel, nrow, ncol};

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, out_dims, ndim + 1, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, grayscale_to_rgb4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}