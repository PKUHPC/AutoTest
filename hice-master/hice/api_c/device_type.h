#pragma once

#include "hice/api_c/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HICE_API_C {
  CPU = 0,
  CUDA = 1,
  NumDeviceTypes = 2,
} HI_DeviceType;

typedef struct {
  HI_DeviceType type;
  int id; /**< The device-id among this specific device type */
} HI_Device;

HICE_API_C extern const HI_Device HI_kCPU;
HICE_API_C extern const HI_Device HI_kCUDA;
HICE_API_C extern const HI_Device HI_kNumDeviceTypes;

#ifdef __cplusplus
}
#endif