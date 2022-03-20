#pragma once

#include "hice/api_c/status.h"
#include "hice/api_c/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Activation functions
 */
HICE_API_C HI_Status HI_Abs(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Relu(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Sigmoid(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Sqrt(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Square(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Tanh(const HI_Tensor input, HI_Tensor* output);
HICE_API_C HI_Status HI_Elu(const HI_Tensor input, const float alpha,
                            HI_Tensor* output);

/**
 * Conv
 */
HICE_API_C HI_Status HI_Conv(const HI_Tensor input, const HI_Tensor kernel,
                             const int* stride, const int stride_len,
                             const int* padding, const int padding_len,
                             const int* dilation, const int dilation_len,
                             const int group_count, HI_Tensor* output);

#ifdef __cplusplus
}
#endif