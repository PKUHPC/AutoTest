#pragma once

#include "hice/api_c/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HICE_API_C HI_LayoutType {
  Dense,
  COO,
  CSR,
  Invalid,
  NumLayoutTypes
} HI_LayoutType;

HICE_API_C extern const HI_LayoutType HI_kDense;
HICE_API_C extern const HI_LayoutType HI_kCOO;
HICE_API_C extern const HI_LayoutType HI_kCSR;
HICE_API_C extern const HI_LayoutType HI_kInvalid;
HICE_API_C extern const HI_LayoutType HI_kNumLayoutTypes;

#ifdef __cplusplus
}
#endif