#include "src/new_ops3/image_normalize.h"
#include "src/core/dispatch.h"

#define image_normalize3d_kernel(typename)                                                               \
    typename *in_data = aitisa_tensor_data(input);                                                       \
    typename *out_data = aitisa_tensor_data(*output);                                                    \
    int64_t channel_offset = ncol * nrow;                                                                \
    for (int64_t i = 0; i < nrow; i++)                                                                   \
    {                                                                                                    \
        for (int64_t j = 0; j < ncol; j++)                                                               \
        {                                                                                                \
            int64_t idx = i * ncol + j;                                                                  \
            out_data[idx] = (in_data[idx] - mean[0]) / std[0];                                           \
            out_data[channel_offset * 1 + idx] = (in_data[channel_offset * 1 + idx] - mean[1]) / std[1]; \
            out_data[channel_offset * 2 + idx] = (in_data[channel_offset * 2 + idx] - mean[2]) / std[2]; \
        }                                                                                                \
    }

#define image_normalize4d_kernel(typename)                                                                   \
    typename *in_data = aitisa_tensor_data(input);                                                           \
    typename *out_data = aitisa_tensor_data(*output);                                                        \
    int64_t channel_offset = ncol * nrow;                                                                    \
    for (int64_t n = 0; n < batch_size; n++)                                                                 \
    {                                                                                                        \
        int64_t batch_offset = n * nchannel * nrow * ncol;                                                   \
        for (int64_t i = 0; i < nrow; i++)                                                                   \
        {                                                                                                    \
            for (int64_t j = 0; j < ncol; j++)                                                               \
            {                                                                                                \
                int64_t idx = i * ncol + j + batch_offset;                                                   \
                out_data[idx] = (in_data[idx] - mean[0]) / std[0];                                           \
                out_data[channel_offset * 1 + idx] = (in_data[channel_offset * 1 + idx] - mean[1]) / std[1]; \
                out_data[channel_offset * 2 + idx] = (in_data[channel_offset * 2 + idx] - mean[2]) / std[2]; \
            }                                                                                                \
        }                                                                                                    \
    }

Status aitisa_image_normalize(const Tensor input, double *mean, double *std, Tensor *output)
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

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, image_normalize3d_kernel);
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

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, image_normalize4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}