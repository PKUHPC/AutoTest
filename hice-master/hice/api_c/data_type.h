#pragma once
#include <stddef.h>
#include "hice/api_c/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  Int8,
  Uint8,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Uint64,
  Float,
  Double,
  Undefined,
  NumScalarTypes
} HI_TypeCode;

/**
 * @brief DataType hold by the tensor
 */
typedef struct {
  HI_TypeCode code; /**< TypeCode used by this DataType*/
  size_t size;  /**< Number of bytes occupyied by this DataType */
} HI_DataType;

HICE_API_C extern const HI_DataType HI_kInt8;
HICE_API_C extern const HI_DataType HI_kUint8;
HICE_API_C extern const HI_DataType HI_kInt16;
HICE_API_C extern const HI_DataType HI_kUint16;
HICE_API_C extern const HI_DataType HI_kInt32;
HICE_API_C extern const HI_DataType HI_kUint32;
HICE_API_C extern const HI_DataType HI_kInt64;
HICE_API_C extern const HI_DataType HI_kUint64;
HICE_API_C extern const HI_DataType HI_kFloat;
HICE_API_C extern const HI_DataType HI_kDouble;
HICE_API_C extern const HI_DataType HI_kUndefined;
HICE_API_C extern const HI_DataType HI_kNumScalarTypes;

#ifdef __cplusplus
}
#endif