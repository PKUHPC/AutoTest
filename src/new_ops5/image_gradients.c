
#include "src/new_ops5/image_gradients.h"
#include "src/core/dispatch.h"

#define transpose3d_kernel(typename)                                                                                           \
    typename *in_data = aitisa_tensor_data(input);                                                                             \
    typename *out_x_data = aitisa_tensor_data(*grad_x);                                                                        \
    typename *out_y_data = aitisa_tensor_data(*grad_y);                                                                        \
    int64_t channel_offset = ncol * nrow;                                                                                      \
    for (int64_t i = 0; i < size; i++)                                                                                         \
    {                                                                                                                          \
        out_x_data[i] = 0.0;                                                                                                   \
        out_y_data[i] = 0.0;                                                                                                   \
    }                                                                                                                          \
    for (int64_t i = 0; i < nrow - 1; i++)                                                                                     \
    {                                                                                                                          \
        for (int64_t j = 0; j < ncol; j++)                                                                                     \
        {                                                                                                                      \
            int64_t idx = i * ncol + j;                                                                                        \
            int64_t idx_next = (i + 1) * ncol + j;                                                                             \
            out_x_data[idx] = in_data[idx_next] - in_data[idx];                                                                \
            out_x_data[idx + channel_offset * 1] = in_data[idx_next + channel_offset * 1] - in_data[idx + channel_offset * 1]; \
            out_x_data[idx + channel_offset * 2] = in_data[idx_next + channel_offset * 2] - in_data[idx + channel_offset * 2]; \
        }                                                                                                                      \
    }                                                                                                                          \
    for (int64_t i = 0; i < nrow; i++)                                                                                         \
    {                                                                                                                          \
        for (int64_t j = 0; j < ncol - 1; j++)                                                                                 \
        {                                                                                                                      \
            int64_t idx = i * ncol + j;                                                                                        \
            int64_t idx_next = i * ncol + j + 1;                                                                               \
            out_y_data[idx] = in_data[idx_next] - in_data[idx];                                                                \
            out_y_data[idx + channel_offset * 1] = in_data[idx_next + channel_offset * 1] - in_data[idx + channel_offset * 1]; \
            out_y_data[idx + channel_offset * 2] = in_data[idx_next + channel_offset * 2] - in_data[idx + channel_offset * 2]; \
        }                                                                                                                      \
    }

#define transpose4d_kernel(typename)                                                                                               \
    typename *in_data = aitisa_tensor_data(input);                                                                                 \
    typename *out_x_data = aitisa_tensor_data(*grad_x);                                                                            \
    typename *out_y_data = aitisa_tensor_data(*grad_y);                                                                            \
    int64_t channel_offset = ncol * nrow;                                                                                          \
    for (int64_t i = 0; i < size; i++)                                                                                             \
    {                                                                                                                              \
        out_x_data[i] = 0.0;                                                                                                       \
        out_y_data[i] = 0.0;                                                                                                       \
    }                                                                                                                              \
    for (int64_t b = 0; b < batch_size; b++)                                                                                       \
    {                                                                                                                              \
        int64_t batch_offset = b * nchannel * ncol * nrow;                                                                         \
        for (int64_t i = 0; i < nrow - 1; i++)                                                                                     \
        {                                                                                                                          \
            for (int64_t j = 0; j < ncol; j++)                                                                                     \
            {                                                                                                                      \
                int64_t idx = i * ncol + j + batch_offset;                                                                         \
                int64_t idx_next = (i + 1) * ncol + j + batch_offset;                                                              \
                out_x_data[idx] = in_data[idx_next] - in_data[idx];                                                                \
                out_x_data[idx + channel_offset * 1] = in_data[idx_next + channel_offset * 1] - in_data[idx + channel_offset * 1]; \
                out_x_data[idx + channel_offset * 2] = in_data[idx_next + channel_offset * 2] - in_data[idx + channel_offset * 2]; \
            }                                                                                                                      \
        }                                                                                                                          \
        for (int64_t i = 0; i < nrow; i++)                                                                                         \
        {                                                                                                                          \
            for (int64_t j = 0; j < ncol - 1; j++)                                                                                 \
            {                                                                                                                      \
                int64_t idx = i * ncol + j + batch_offset;                                                                         \
                int64_t idx_next = i * ncol + j + 1 + batch_offset;                                                                \
                out_y_data[idx] = in_data[idx_next] - in_data[idx];                                                                \
                out_y_data[idx + channel_offset * 1] = in_data[idx_next + channel_offset * 1] - in_data[idx + channel_offset * 1]; \
                out_y_data[idx + channel_offset * 2] = in_data[idx_next + channel_offset * 2] - in_data[idx + channel_offset * 2]; \
            }                                                                                                                      \
        }                                                                                                                          \
    }

    Status aitisa_image_gradients(const Tensor input, Tensor *grad_x, Tensor *grad_y)
    {
        int64_t *dims = aitisa_tensor_dims(input);
        int64_t ndim = aitisa_tensor_ndim(input);
        Status status = STATUS_SUCCESS;
        if (ndim == 3)
        {
            int nchannel = dims[0];
            int nrow = dims[1];
            int ncol = dims[2];

            if (nchannel != 3)
            {
                status = STATUS_TYPE_MISMATCH;
                return status;
            }

            Tensor new_tensor1;
            Tensor new_tensor2;
            DataType dtype = aitisa_tensor_data_type(input);
            Device device = aitisa_tensor_device(input);

            CHECK_STATUS(
                aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor1));
            CHECK_STATUS(
                aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor2));
            *grad_x = new_tensor1;
            *grad_y = new_tensor2;
            int64_t size = aitisa_tensor_size(input);
            AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, transpose3d_kernel);
        }
        else if (ndim == 4)
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

            Tensor new_tensor1;
            Tensor new_tensor2;
            DataType dtype = aitisa_tensor_data_type(input);
            Device device = aitisa_tensor_device(input);

            CHECK_STATUS(
                aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor1));
            CHECK_STATUS(
                aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor2));
            *grad_x = new_tensor1;
            *grad_y = new_tensor2;
            int64_t size = aitisa_tensor_size(input);
            AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, transpose4d_kernel);
        }
        else
        {
            status = STATUS_TYPE_MISMATCH;
        }
        return status;
    }