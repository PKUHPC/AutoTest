#include "src/new_ops5/gaussian_blur.h"
#include "src/core/dispatch.h"
#include "math.h"

double PI = 3.14159265359;

// double v = 1.0 / (2 * PI * sigma * sigma) * exp(-1.0 / (2 * sigma * sigma) * (x * x + y * y));                                 \

#define gaussian_blur3d_kernel(typename)                                                                                                            \
    typename *in_data = aitisa_tensor_data(input);                                                                                                  \
    typename *out_data = aitisa_tensor_data(*output);                                                                                               \
    typename *kernel_data = aitisa_tensor_data(kernel);                                                                                             \
    int64_t in_channel_offset = ncol * nrow;                                                                                                        \
    int64_t radius = kernel_size / 2;                                                                                                               \
    double kernel_sum = 0.0;                                                                                                                        \
    for (int64_t x = -radius; x <= radius; x++)                                                                                                     \
    {                                                                                                                                               \
        for (int64_t y = -radius; y <= radius; y++)                                                                                                 \
        {                                                                                                                                           \
            double v = 1.0;                                                                                                                         \
            int64_t kernel_offset = (x + radius) * kernel_size + y + radius;                                                                        \
            kernel_data[kernel_offset] = v;                                                                                                         \
            kernel_sum += v;                                                                                                                        \
        }                                                                                                                                           \
    }                                                                                                                                               \
    for (int64_t x = -radius; x <= radius; x++)                                                                                                     \
    {                                                                                                                                               \
        for (int64_t y = -radius; y <= radius; y++)                                                                                                 \
        {                                                                                                                                           \
            int64_t kernel_offset = (x + radius) * kernel_size + y + radius;                                                                        \
            kernel_data[kernel_offset] /= kernel_sum;                                                                                               \
        }                                                                                                                                           \
    }                                                                                                                                               \
    for (int64_t i = 0; i < nrow; i++)                                                                                                              \
    {                                                                                                                                               \
        for (int64_t j = 0; j < ncol; j++)                                                                                                          \
        {                                                                                                                                           \
            double sumR = 0.0;                                                                                                                      \
            double sumG = 0.0;                                                                                                                      \
            double sumB = 0.0;                                                                                                                      \
            int64_t idx = i * ncol + j;                                                                                                             \
            for (int64_t ki = -radius; ki <= radius; ki++)                                                                                          \
            {                                                                                                                                       \
                for (int64_t kj = -radius; kj <= radius; kj++)                                                                                      \
                {                                                                                                                                   \
                    if (i + ki < 0 || i + ki >= nrow || j + kj < 0 || j + kj >= ncol)                                                               \
                    {                                                                                                                               \
                        sumR += 0;                                                                                                                  \
                        sumG += 0;                                                                                                                  \
                        sumB += 0;                                                                                                                  \
                    }                                                                                                                               \
                    else                                                                                                                            \
                    {                                                                                                                               \
                        sumR += in_data[(i + ki) * ncol + j + kj + in_channel_offset * 0] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                        sumG += in_data[(i + ki) * ncol + j + kj + in_channel_offset * 1] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                        sumB += in_data[(i + ki) * ncol + j + kj + in_channel_offset * 2] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                    }                                                                                                                               \
                }                                                                                                                                   \
            }                                                                                                                                       \
            out_data[idx + in_channel_offset * 0] = sumR;                                                                                           \
            out_data[idx + in_channel_offset * 1] = sumG;                                                                                           \
            out_data[idx + in_channel_offset * 2] = sumB;                                                                                           \
        }                                                                                                                                           \
    }

#define gaussian_blur4d_kernel(typename)                                                                                                                                             \
    typename *in_data = aitisa_tensor_data(input);                                                                                                                                   \
    typename *out_data = aitisa_tensor_data(*output);                                                                                                                                \
    typename *kernel_data = aitisa_tensor_data(kernel);                                                                                                                              \
    int64_t in_channel_offset = ncol * nrow;                                                                                                                                         \
    int64_t radius = kernel_size / 2;                                                                                                                                                \
    double kernel_sum = 0.0;                                                                                                                                                         \
    for (int64_t x = -radius; x <= radius; x++)                                                                                                                                      \
    {                                                                                                                                                                                \
        for (int64_t y = -radius; y <= radius; y++)                                                                                                                                  \
        {                                                                                                                                                                            \
            double v = 1.0;                                                                                                                                                          \
            int64_t kernel_offset = (x + radius) * kernel_size + y + radius;                                                                                                         \
            kernel_data[kernel_offset] = v;                                                                                                                                          \
            kernel_sum += v;                                                                                                                                                         \
        }                                                                                                                                                                            \
    }                                                                                                                                                                                \
    for (int64_t x = -radius; x <= radius; x++)                                                                                                                                      \
    {                                                                                                                                                                                \
        for (int64_t y = -radius; y <= radius; y++)                                                                                                                                  \
        {                                                                                                                                                                            \
            int64_t kernel_offset = (x + radius) * kernel_size + y + radius;                                                                                                         \
            kernel_data[kernel_offset] /= kernel_sum;                                                                                                                                \
        }                                                                                                                                                                            \
    }                                                                                                                                                                                \
    for (int64_t n = 0; n < batch_size; n++)                                                                                                                                         \
    {                                                                                                                                                                                \
        for (int64_t i = 0; i < nrow; i++)                                                                                                                                           \
        {                                                                                                                                                                            \
            for (int64_t j = 0; j < ncol; j++)                                                                                                                                       \
            {                                                                                                                                                                        \
                double sumR = 0.0;                                                                                                                                                   \
                double sumG = 0.0;                                                                                                                                                   \
                double sumB = 0.0;                                                                                                                                                   \
                int64_t idx = i * ncol + j + n * nchannel * ncol * nrow;                                                                                                             \
                for (int64_t ki = -radius; ki <= radius; ki++)                                                                                                                       \
                {                                                                                                                                                                    \
                    for (int64_t kj = -radius; kj <= radius; kj++)                                                                                                                   \
                    {                                                                                                                                                                \
                        if (i + ki < 0 || i + ki >= nrow || j + kj < 0 || j + kj >= ncol)                                                                                            \
                        {                                                                                                                                                            \
                            sumR += 0;                                                                                                                                               \
                            sumG += 0;                                                                                                                                               \
                            sumB += 0;                                                                                                                                               \
                        }                                                                                                                                                            \
                        else                                                                                                                                                         \
                        {                                                                                                                                                            \
                            sumR += in_data[n * nchannel * ncol * nrow + (i + ki) * ncol + j + kj + in_channel_offset * 0] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                            sumG += in_data[n * nchannel * ncol * nrow + (i + ki) * ncol + j + kj + in_channel_offset * 1] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                            sumB += in_data[n * nchannel * ncol * nrow + (i + ki) * ncol + j + kj + in_channel_offset * 2] * kernel_data[(ki + radius) * kernel_size + kj + radius]; \
                        }                                                                                                                                                            \
                    }                                                                                                                                                                \
                }                                                                                                                                                                    \
                out_data[idx + in_channel_offset * 0] = sumR;                                                                                                                        \
                out_data[idx + in_channel_offset * 1] = sumG;                                                                                                                        \
                out_data[idx + in_channel_offset * 2] = sumB;                                                                                                                        \
            }                                                                                                                                                                        \
        }                                                                                                                                                                            \
    }
Status aitisa_gaussian_blur(const Tensor input, const int kernel_size, const double sigma, Tensor *output)
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

        int64_t *kernel_shape[2] = {kernel_size, kernel_size};

        Tensor new_tensor;
        Tensor kernel;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        CHECK_STATUS(
            aitisa_create(dtype, device, kernel_shape, 2, NULL, 0, &kernel));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, gaussian_blur3d_kernel);
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

        int64_t *kernel_shape[2] = {kernel_size, kernel_size};

        Tensor new_tensor;
        Tensor kernel;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        CHECK_STATUS(
            aitisa_create(dtype, device, kernel_shape, 2, NULL, 0, &kernel));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, gaussian_blur4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}