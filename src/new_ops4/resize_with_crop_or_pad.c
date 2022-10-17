#include "src/new_ops4/resize_with_crop_or_pad.h"
#include "src/core/dispatch.h"

#define resize_with_crop_or_pad3d_kernel(typename)                                                \
    typename *in_data = aitisa_tensor_data(input);                                                \
    typename *out_data = aitisa_tensor_data(*output);                                             \
    int64_t in_channel_offset = ncol * nrow;                                                      \
    int64_t out_channel_offset = target_h * target_w;                                             \
    for (int64_t i = 0; i < h; i++)                                                               \
    {                                                                                             \
        for (int64_t j = 0; j < w; j++)                                                           \
        {                                                                                         \
            int64_t in_idx = (start_i_in + i) * ncol + (start_j_in + j);                          \
            int64_t out_idx = (start_i_out + i) * target_w + (start_j_out + j);                   \
            out_data[out_idx] = in_data[in_idx];                                                  \
            out_data[out_idx + out_channel_offset * 1] = in_data[in_idx + in_channel_offset * 1]; \
            out_data[out_idx + out_channel_offset * 2] = in_data[in_idx + in_channel_offset * 2]; \
        }                                                                                         \
    }

#define resize_with_crop_or_pad4d_kernel(typename)                                                     \
    typename *in_data = aitisa_tensor_data(input);                                                     \
    typename *out_data = aitisa_tensor_data(*output);                                                  \
    int64_t in_channel_offset = ncol * nrow;                                                           \
    int64_t out_channel_offset = target_h * target_w;                                                  \
    for (int64_t b = 0; b < batch_size; b++)                                                           \
    {                                                                                                  \
        int64_t in_batch_offset = b * nchannel * nrow * ncol;                                          \
        int64_t out_batch_offset = b * nchannel * target_h * target_w;                                 \
        for (int64_t i = 0; i < target_h; i++)                                                         \
        {                                                                                              \
            for (int64_t j = 0; j < target_w; j++)                                                     \
            {                                                                                          \
                int64_t in_idx = (start_i_in + i) * ncol + (start_j_in + j) + in_batch_offset;         \
                int64_t out_idx = (start_i_out + i) * target_w + (start_j_out + j) + out_batch_offset; \
                out_data[out_idx] = in_data[in_idx];                                                   \
                out_data[out_idx + out_channel_offset * 1] = in_data[in_idx + in_channel_offset * 1];  \
                out_data[out_idx + out_channel_offset * 2] = in_data[in_idx + in_channel_offset * 2];  \
            }                                                                                          \
        }                                                                                              \
    }

Status aitisa_resize_with_crop_or_pad(const Tensor input, int target_h, int target_w, Tensor *output)
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

        int64_t *new_dims[3] = {nchannel, target_h, target_w};
        int64_t start_i_in = 0;
        int64_t start_i_out = 0;
        int64_t start_j_in = 0;
        int64_t start_j_out = 0;
        int64_t h = 0;
        int64_t w = 0;
        if (nrow >= target_h)
        {
            start_i_in = (nrow - target_h) / 2;
            start_i_out = 0;
            h = target_h;
        }
        else
        {
            start_i_in = 0;
            start_i_out = (target_h - nrow) / 2;
            h = nrow;
        }

        if (ncol >= target_w)
        {
            start_j_in = (ncol - target_w) / 2;
            start_j_out = 0;
            w = target_w;
        }
        else
        {
            start_j_in = 0;
            start_j_out = (target_w - ncol) / 2;
            w = ncol;
        }

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, resize_with_crop_or_pad3d_kernel);
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
        int64_t start_i_in = 0;
        int64_t start_i_out = 0;
        int64_t start_j_in = 0;
        int64_t start_j_out = 0;
        if (nrow >= target_h)
        {
            start_i_in = (nrow - target_h) / 2;
            start_i_out = 0;
        }
        else
        {
            start_i_in = 0;
            start_i_out = (target_h - nrow) / 2;
        }

        if (ncol >= target_w)
        {
            start_j_in = (ncol - target_w) / 2;
            start_j_out = 0;
        }
        else
        {
            start_j_in = 0;
            start_j_out = (target_w - ncol) / 2;
        }

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, new_dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, resize_with_crop_or_pad4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}