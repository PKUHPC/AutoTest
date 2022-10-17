#include "src/new_ops2/hsv_to_rgb.h"
#include "src/core/dispatch.h"
#include "math.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define hsv_to_rgb3d_kernel(typename)                                                            \
    typename *in_data = aitisa_tensor_data(input);                                               \
    typename *out_data = aitisa_tensor_data(*output);                                            \
    int64_t channel_offset = ncol * nrow;                                                        \
    for (int64_t i = 0; i < nrow; i++)                                                           \
    {                                                                                            \
        for (int64_t j = 0; j < ncol; j++)                                                       \
        {                                                                                        \
            int64_t idx = i * ncol + j;                                                          \
            double H = in_data[channel_offset * 0 + idx];                                        \
            double S = in_data[channel_offset * 1 + idx];                                        \
            double V = in_data[channel_offset * 2 + idx];                                        \
            double C = V * S;                                                                    \
            double X = C * (1 - abs(((int)H / 60) % 2 - 1));                                     \
            double m = V - C;                                                                    \
            if (H >= 0 && H < 60)                                                                \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = C;                                          \
                out_data[channel_offset * 1 + idx] = X;                                          \
                out_data[channel_offset * 2 + idx] = 0;                                          \
            }                                                                                    \
            else if (H >= 60 && H < 120)                                                         \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = X;                                          \
                out_data[channel_offset * 1 + idx] = C;                                          \
                out_data[channel_offset * 2 + idx] = 0;                                          \
            }                                                                                    \
            else if (H >= 120 && H < 180)                                                        \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = 0;                                          \
                out_data[channel_offset * 1 + idx] = C;                                          \
                out_data[channel_offset * 2 + idx] = X;                                          \
            }                                                                                    \
            else if (H >= 180 && H < 240)                                                        \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = 0;                                          \
                out_data[channel_offset * 1 + idx] = X;                                          \
                out_data[channel_offset * 2 + idx] = C;                                          \
            }                                                                                    \
            else if (H >= 240 && H < 300)                                                        \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = X;                                          \
                out_data[channel_offset * 1 + idx] = 0;                                          \
                out_data[channel_offset * 2 + idx] = C;                                          \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                out_data[channel_offset * 0 + idx] = C;                                          \
                out_data[channel_offset * 1 + idx] = 0;                                          \
                out_data[channel_offset * 2 + idx] = X;                                          \
            }                                                                                    \
            out_data[channel_offset * 0 + idx] = (out_data[channel_offset * 0 + idx] + m) * 255; \
            out_data[channel_offset * 1 + idx] = (out_data[channel_offset * 1 + idx] + m) * 255; \
            out_data[channel_offset * 2 + idx] = (out_data[channel_offset * 2 + idx] + m) * 255; \
        }                                                                                        \
    }

#define hsv_to_rgb4d_kernel(typename)                                                                \
    typename *in_data = aitisa_tensor_data(input);                                                   \
    typename *out_data = aitisa_tensor_data(*output);                                                \
    int64_t channel_offset = ncol * nrow;                                                            \
    for (int64_t b = 0; b < batch_size; b++)                                                         \
    {                                                                                                \
        int64_t batch_offset = b * nchannel * nrow * ncol;                                           \
        for (int64_t i = 0; i < nrow; i++)                                                           \
        {                                                                                            \
            for (int64_t j = 0; j < ncol; j++)                                                       \
            {                                                                                        \
                int64_t idx = batch_offset + i * ncol + j;                                           \
                double H = in_data[channel_offset * 0 + idx];                                        \
                double S = in_data[channel_offset * 1 + idx];                                        \
                double V = in_data[channel_offset * 2 + idx];                                        \
                double C = V * S;                                                                    \
                double X = C * (1 - abs(((int)H / 60) % 2 - 1));                                     \
                double m = V - C;                                                                    \
                if (H >= 0 && H < 60)                                                                \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = C;                                          \
                    out_data[channel_offset * 1 + idx] = X;                                          \
                    out_data[channel_offset * 2 + idx] = 0;                                          \
                }                                                                                    \
                else if (H >= 60 && H < 120)                                                         \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = X;                                          \
                    out_data[channel_offset * 1 + idx] = C;                                          \
                    out_data[channel_offset * 2 + idx] = 0;                                          \
                }                                                                                    \
                else if (H >= 120 && H < 180)                                                        \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = 0;                                          \
                    out_data[channel_offset * 1 + idx] = C;                                          \
                    out_data[channel_offset * 2 + idx] = X;                                          \
                }                                                                                    \
                else if (H >= 180 && H < 240)                                                        \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = 0;                                          \
                    out_data[channel_offset * 1 + idx] = C;                                          \
                    out_data[channel_offset * 2 + idx] = X;                                          \
                }                                                                                    \
                else if (H >= 240 && H < 300)                                                        \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = X;                                          \
                    out_data[channel_offset * 1 + idx] = 0;                                          \
                    out_data[channel_offset * 2 + idx] = C;                                          \
                }                                                                                    \
                else                                                                                 \
                {                                                                                    \
                    out_data[channel_offset * 0 + idx] = C;                                          \
                    out_data[channel_offset * 1 + idx] = 0;                                          \
                    out_data[channel_offset * 2 + idx] = X;                                          \
                }                                                                                    \
                out_data[channel_offset * 0 + idx] = (out_data[channel_offset * 0 + idx] + m) * 255; \
                out_data[channel_offset * 1 + idx] = (out_data[channel_offset * 1 + idx] + m) * 255; \
                out_data[channel_offset * 2 + idx] = (out_data[channel_offset * 2 + idx] + m) * 255; \
            }                                                                                        \
        }                                                                                            \
    }

Status aitisa_hsv_to_rgb(const Tensor input, Tensor *output)
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, hsv_to_rgb3d_kernel);
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, hsv_to_rgb4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}