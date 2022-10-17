#include "src/new_ops1/vertical_flip.h"
#include "src/core/dispatch.h"

#define vflip2d_kernel(typename)                             \
    typename *in_data = aitisa_tensor_data(input);           \
    typename *out_data = aitisa_tensor_data(*output);        \
    for (int64_t i = 0; i < nrow; i++)                       \
    {                                                        \
        for (int64_t j = 0; j < ncol; j++)                   \
        {                                                    \
            int64_t out_idx = i * ncol + j;                  \
            int64_t in_idx = (nrow - 1 - i) * ncol + j;      \
            out_data[out_idx] = (typename)(in_data[in_idx]); \
        }                                                    \
    }

#define vflip3d_kernel(typename)                                             \
    typename *in_data = aitisa_tensor_data(input);                           \
    typename *out_data = aitisa_tensor_data(*output);                        \
    for (int64_t c = 0; c < nchannel; c++)                                   \
    {                                                                        \
        int64_t channel_offset = ncol * nrow * c;                            \
        for (int64_t i = 0; i < nrow; i++)                                   \
        {                                                                    \
            for (int64_t j = 0; j < ncol; j++)                               \
            {                                                                \
                int64_t out_idx = channel_offset + i * ncol + j;             \
                int64_t in_idx = channel_offset + (nrow - 1 - i) * ncol + j; \
                out_data[out_idx] = (typename)(in_data[in_idx]);             \
            }                                                                \
        }                                                                    \
    }

Status aitisa_vertical_flip(const Tensor input, Tensor *output)
{
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    if(ndim==2)
    {
        int nrow = dims[0];
        int ncol = dims[1];

        Tensor new_tensor;
        DataType dtype = aitisa_tensor_data_type(input);
        Device device = aitisa_tensor_device(input);

        CHECK_STATUS(
            aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
        *output = new_tensor;
        int64_t size = aitisa_tensor_size(input);
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, vflip2d_kernel);
    }
    else if(ndim==3)
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, vflip3d_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}