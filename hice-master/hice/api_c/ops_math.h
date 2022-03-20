#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "hice/api_c/tensor.h"
#include "hice/api_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { SUM = 0, PROD, MAX, MIN, MEAN } HI_ReduceMode;

HICE_API_C HI_Status HI_Matmul(const HI_Tensor tensor1, const HI_Tensor tensor2,
                               HI_Tensor *output);

HICE_API_C HI_Status HI_Reduce(const HI_Tensor input, HI_ReduceMode mode,
                               const int64_t *axis, int64_t axis_len,
                               bool keepdim, HI_Tensor *output);

HICE_API_C HI_Status HI_Add(const HI_Tensor tensor1, const HI_Tensor tensor2,
                            HI_Tensor *output);
HICE_API_C HI_Status HI_Sub(const HI_Tensor tensor1, const HI_Tensor tensor2,
                            HI_Tensor *output);
HICE_API_C HI_Status HI_Mul(const HI_Tensor tensor1, const HI_Tensor tensor2,
                            HI_Tensor *output);
HICE_API_C HI_Status HI_Div(const HI_Tensor tensor1, const HI_Tensor tensor2,
                            HI_Tensor *output);

HICE_API_C HI_Status HI_Exp(const HI_Tensor input, HI_Tensor *output);
HICE_API_C HI_Status HI_Log(const HI_Tensor input, HI_Tensor *output);
HICE_API_C HI_Status HI_Neg(const HI_Tensor input, HI_Tensor *output);

HICE_API_C HI_Status HI_Equal(const HI_Tensor tensor1, const HI_Tensor tensor2,
                              HI_Tensor *output);
HICE_API_C HI_Status HI_Less(const HI_Tensor tensor1, const HI_Tensor tensor2,
                             HI_Tensor *output);
HICE_API_C HI_Status HI_LessEqual(const HI_Tensor tensor1,
                                  const HI_Tensor tensor2, HI_Tensor *output);
HICE_API_C HI_Status HI_Greater(const HI_Tensor tensor1,
                                const HI_Tensor tensor2, HI_Tensor *output);
HICE_API_C HI_Status HI_GreaterEqual(const HI_Tensor tensor1,
                                     const HI_Tensor tensor2,
                                     HI_Tensor *output);
#if 1
// inplace
HICE_API_C
HI_Status HI_Matmul_Inplace(const HI_Tensor tensor1, const HI_Tensor tensor2,
                            HI_Tensor output);
#endif

#ifdef __cplusplus
}
#endif