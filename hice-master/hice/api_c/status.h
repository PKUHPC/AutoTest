#pragma once

#include "hice/api_c/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  Success = 0,
  TypeMismatch,
  DimensionsMismatch,
  InvalidArgument,
  NotSupported,
  AllocFailed,
  OutOfRange,
  InternalError,
  MathError,
  UnknownError
} HI_Status;

HICE_API_C void HI_CheckStatus(HI_Status status);

HICE_API_C const char* HI_GetLastError();

HICE_API_C HI_Status Str2Status(const char* );

HICE_API_C const char* Status2Str(HI_Status status);

#ifdef __cplusplus
}
#endif