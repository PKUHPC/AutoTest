#include "src/new_ops3/adjust_contrast.h"
#include "src/core/dispatch.h"

#define adjust_contrast3d_kernel(typename)                                                                               \
    typename *in_data = aitisa_tensor_data(input);                                                                              \
    typename *out_data = aitisa_tensor_data(*output);                                                                           \
    double sum = 0.0;                                                                                                           \
    for (int64_t c = 0; c < nchannel; c++)                                                                                      \
    {                                                                                                                           \
        double sum = 0.0;                                                                                                       \
        int64_t channel_offset = c * ncol * nrow;                                                                               \
        for (int64_t i = 0; i < ncol * nrow; i++)                                                                               \
        {                                                                                                                       \
            sum += in_data[i + channel_offset];                                                                                 \
        }                                                                                                                       \
        sum /= (ncol * nrow);                                                                                                   \
        for (int64_t i = 0; i < ncol * nrow; i++)                                                                               \
        {                                                                                                                       \
            out_data[i + channel_offset] = (in_data[i + channel_offset] - sum) * contrast_factor + in_data[i + channel_offset]; \
            if (out_data[i + channel_offset] < 0)                                                                               \
            {                                                                                                                   \
                out_data[i + channel_offset] = 0;                                                                               \
            }                                                                                                                   \
            if (out_data[i + channel_offset] > 1)                                                                               \
            {                                                                                                                   \
                out_data[i + channel_offset] = 1;                                                                               \
            }                                                                                                                   \
        }                                                                                                                       \
    }

#define adjust_contrast4d_kernel(typename)                                                                                   \
    typename *in_data = aitisa_tensor_data(input);                                                                                  \
    typename *out_data = aitisa_tensor_data(*output);                                                                               \
    double sum = 0.0;                                                                                                               \
    for (int64_t n = 0; n < batch_size; n++)                                                                                        \
    {                                                                                                                               \
        int64_t batch_offset = n * nchannel * nrow * ncol;                                                                          \
        for (int64_t c = 0; c < nchannel; c++)                                                                                      \
        {                                                                                                                           \
            double sum = 0.0;                                                                                                       \
            int64_t channel_offset = c * ncol * nrow + batch_offset;                                                                \
            for (int64_t i = 0; i < ncol * nrow; i++)                                                                               \
            {                                                                                                                       \
                sum += in_data[i + channel_offset];                                                                                 \
            }                                                                                                                       \
            sum /= (ncol * nrow);                                                                                                   \
            for (int64_t i = 0; i < ncol * nrow; i++)                                                                               \
            {                                                                                                                       \
                out_data[i + channel_offset] = (in_data[i + channel_offset] - sum) * contrast_factor + in_data[i + channel_offset]; \
                if (out_data[i + channel_offset] < 0)                                                                               \
                {                                                                                                                   \
                    out_data[i + channel_offset] = 0;                                                                               \
                }                                                                                                                   \
                if (out_data[i + channel_offset] > 1)                                                                               \
                {                                                                                                                   \
                    out_data[i + channel_offset] = 1;                                                                               \
                }                                                                                                                   \
            }                                                                                                                       \
        }                                                                                                                           \
    }

Status aitisa_adjust_contrast(const Tensor input, double contrast_factor, Tensor *output)
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, adjust_contrast3d_kernel);
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, adjust_contrast4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}