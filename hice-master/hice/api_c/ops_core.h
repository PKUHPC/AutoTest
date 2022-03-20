#pragma once

#include <stdint.h>
#include "hice/api_c/tensor.h"
#include "hice/api_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

HICE_API_C HI_Status HI_TensorSize(const HI_Tensor tensor1, int64_t *size);

HICE_API_C HI_Status HI_TensorItemSize(const HI_Tensor tensor1, size_t *item_size);

HICE_API_C HI_Status HI_TensorDims(const HI_Tensor tensor1, const int64_t **dims);

HICE_API_C HI_Status HI_TensorNdim(const HI_Tensor tensor1, int64_t *ndim);

HICE_API_C HI_Status HI_TensorRawMutableData(HI_Tensor tensor1, void *raw_data);

HICE_API_C HI_Status HI_Print(const HI_Tensor tensor1);

/**
 * @brief transport tensor from one device to another. 
 */
HICE_API_C HI_Status HI_ToDevice(HI_Tensor input, HI_Device hi_device,
                                 HI_Tensor *output);

#ifdef __cplusplus
}
#endif