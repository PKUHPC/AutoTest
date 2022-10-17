#include "src/new_ops3/random_brightness.h"
#include "src/core/dispatch.h"
#include "time.h"
#include <stdlib.h>

#define random_brightness_kernel(typename)     \
    typename *in_data = aitisa_tensor_data(input);    \
    typename *out_data = aitisa_tensor_data(*output); \
    for (int64_t i = 0; i < size; i++)                \
    {                                                 \
        out_data[i] = in_data[i] + brightness_factor; \
        if (out_data[i] < 0)                          \
        {                                             \
            out_data[i] = 0;                          \
        }                                             \
        if (out_data[i] > 1)                          \
        {                                             \
            out_data[i] = 1;                          \
        }                                             \
    }

Status aitisa_random_brightness(const Tensor input, double lower, double upper, Tensor *output)
{
    int64_t *dims = aitisa_tensor_dims(input);
    int64_t ndim = aitisa_tensor_ndim(input);
    Status status = STATUS_SUCCESS;
    srand(0);
    double brightness_factor = (rand() / (double)RAND_MAX) * (upper - lower) + lower;
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_brightness_kernel);
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
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, random_brightness_kernel);
    }
    else
    {
        status = STATUS_TYPE_MISMATCH;
    }
    return status;
}