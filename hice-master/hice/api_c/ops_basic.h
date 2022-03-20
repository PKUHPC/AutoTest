#pragma once

#include <stdint.h>
#include "hice/api_c/status.h"
#include "hice/api_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief create a dense tensor with 'data'. (do copy)
 * @param len length of data in bytes
 */
HICE_API_C HI_Status HI_Create(HI_DataType hi_data_type, HI_Device hi_device,
                               int64_t *dims, int64_t ndim, void *data,
                               size_t len, HI_Tensor *output);

/**
 * @brief create a dense tensor with outside 'data'. (no copy)
 * @param len length of data in bytes
 */
HICE_API_C HI_Status HI_Wrap(HI_DataType hi_data_type, HI_Device hi_device,
                             int64_t *dims, int64_t ndim, void *data,
                             HI_Tensor *output);

/**
 * @brief create a dense tensor filled with specific value.
 */
HICE_API_C HI_Status HI_Full(HI_DataType hi_data_type, HI_Device hi_device,
                             int64_t *dims, int64_t ndim, void *value,
                             HI_Tensor *output);

/**
 * @brief create a dense tensor filled with random value meeting the uniform
 * distribution.
 */
HICE_API_C HI_Status HI_RandUniform(HI_DataType hi_data_type,
                                    HI_Device hi_device, int64_t *dims,
                                    int64_t ndim, void *a, void *b,
                                    HI_Tensor *output);

/**
 * @brief create a dense tensor filled with random value meeting the normal
 * distribution.
 */
HICE_API_C HI_Status HI_RandNormal(HI_DataType hi_data_type,
                                   HI_Device hi_device, int64_t *dims,
                                   int64_t ndim, void *mean, void *stddev,
                                   HI_Tensor *output);
#ifdef __cplusplus
}
#endif