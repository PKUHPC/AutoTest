#include "src/new_ops1/random_horizontal_flip.h"
#include "src/core/dispatch.h"

#define random_hflip3d_kernel(typename)                                      \
    typename *in_data = aitisa_tensor_data(input);                           \
    typename *out_data = aitisa_tensor_data(*output);                        \
    int random_num = rand() % 100 + 1;                                       \
    for (int64_t c = 0; c < nchannel; c++)                                   \
    {                                                                        \
        int64_t channel_offset = ncol * nrow * c;                            \
        for (int64_t i = 0; i < nrow; i++)                                   \
        {                                                                    \
            for (int64_t j = 0; j < ncol; j++)                               \
            {                                                                \
                int64_t out_idx = channel_offset + i * ncol + j;             \
                int64_t in_idx = channel_offset + i * ncol + (ncol - 1 - j); \
                if (random_num < rate_line)                                  \
                {                                                            \
                    out_data[out_idx] = (typename)(in_data[in_idx]);         \
                }                                                            \
                else                                                         \
                {                                                            \
                    out_data[out_idx] = (typename)(in_data[out_idx]);        \
                }                                                            \
            }                                                                \
        }                                                                    \
    }

#define random_hflip4d_kernel(typename)                                            \
    typename *in_data = aitisa_tensor_data(input);                                 \
    typename *out_data = aitisa_tensor_data(*output);                              \
    for (int64_t b = 0; b < batch_size; b++)                                       \
    {                                                                              \
        int random_num = rand() % 100 + 1;                                         \
        for (int64_t c = 0; c < nchannel; c++)                                     \
        {                                                                          \
            int64_t channel_offset = b * nchannel * ncol * nrow + ncol * nrow * c; \
            for (int64_t i = 0; i < nrow; i++)                                     \
            {                                                                      \
                for (int64_t j = 0; j < ncol; j++)                                 \
                {                                                                  \
                    int64_t out_idx = channel_offset + i * ncol + j;               \
                    int64_t in_idx = channel_offset + i * ncol + (ncol - 1 - j);   \
                    if (random_num < rate_line)                                    \
                    {                                                              \
                        out_data[out_idx] = (typename)(in_data[in_idx]);           \
                    }                                                              \
                    else                                                           \
                    {                                                              \
                        out_data[out_idx] = (typename)(in_data[out_idx]);          \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

Status aitisa_random_horizontal_flip(const Tensor input, const float prob, const int seed, Tensor *output)
{
    // check params
    if (prob < 0 || prob > 1)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    srand(seed);
    int rate_line = (int)(100 * prob);
    if (ndim == 3) // [C, H, W]
    {
        int nchannel = dims[0];
        int nrow = dims[1];
        int ncol = dims[2];

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_hflip3d_kernel);
    }
    else if (ndim == 4) // [N, C, H, W]
    {
        int batch_size = dims[0];
        int nchannel = dims[1];
        int nrow = dims[2];
        int ncol = dims[3];

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_hflip4d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}