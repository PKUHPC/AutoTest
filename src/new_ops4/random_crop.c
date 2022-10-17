#include "src/new_ops4/random_crop.h"
#include "src/core/dispatch.h"

#define random_crop3d_kernel(typename)                                                            \
    typename *in_data = aitisa_tensor_data(input);                                                \
    typename *out_data = aitisa_tensor_data(*output);                                             \
    int64_t in_channel_offset = ncol * nrow;                                                      \
    int64_t out_channel_offset = target_h * target_w;                                             \
    for (int64_t i = 0; i < target_h; i++)                                                        \
    {                                                                                             \
        for (int64_t j = 0; j < target_w; j++)                                                    \
        {                                                                                         \
            int64_t in_idx = (start_i + i) * ncol + (start_j + j);                                \
            int64_t out_idx = i * target_w + j;                                                   \
            out_data[out_idx] = in_data[in_idx];                                                  \
            out_data[out_idx + out_channel_offset * 1] = in_data[in_idx + in_channel_offset * 1]; \
            out_data[out_idx + out_channel_offset * 2] = in_data[in_idx + in_channel_offset * 2]; \
        }                                                                                         \
    }

#define random_crop4d_kernel(typename)                                                                \
    typename *in_data = aitisa_tensor_data(input);                                                    \
    typename *out_data = aitisa_tensor_data(*output);                                                 \
    int64_t in_channel_offset = ncol * nrow;                                                          \
    int64_t out_channel_offset = target_h * target_w;                                                 \
    for (int64_t b = 0; b < batch_size; b++)                                                          \
    {                                                                                                 \
        int64_t in_batch_offset = b * nchannel * nrow * ncol;                                         \
        int64_t out_batch_offset = b * nchannel * target_h * target_w;                                \
        for (int64_t i = 0; i < target_h; i++)                                                        \
        {                                                                                             \
            for (int64_t j = 0; j < target_w; j++)                                                    \
            {                                                                                         \
                int64_t in_idx = (start_i + i) * ncol + (start_j + j) + in_batch_offset;              \
                int64_t out_idx = i * target_w + j + out_batch_offset;                                \
                out_data[out_idx] = in_data[in_idx];                                                  \
                out_data[out_idx + out_channel_offset * 1] = in_data[in_idx + in_channel_offset * 1]; \
                out_data[out_idx + out_channel_offset * 2] = in_data[in_idx + in_channel_offset * 2]; \
            }                                                                                         \
        }                                                                                             \
    }

Status aitisa_random_crop(const Tensor input, int target_h, int target_w, Tensor *output)
{
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    srand(0);
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

        int64_t *new_dims[3] = {nchannel, target_h, target_w};

        int64_t start_i = rand() % (nrow - target_h + 1);
        int64_t start_j = rand() % (ncol - target_w + 1);

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_crop3d_kernel);
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

        int64_t *new_dims[4] = {batch_size, nchannel, target_h, target_w};
        int64_t start_i = rand() % (nrow - target_h + 1);
        int64_t start_j = rand() % (ncol - target_w + 1);

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_crop4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}