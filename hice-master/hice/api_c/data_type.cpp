#include "hice/api_c/data_type.h"
#include <stdint.h>

const HI_DataType HI_kInt8 = {Int8, sizeof(int8_t)};
const HI_DataType HI_kUint8 = {Uint8, sizeof(uint8_t)};
const HI_DataType HI_kInt16 = {Int16, sizeof(int16_t)};
const HI_DataType HI_kUint16 = {Uint16, sizeof(uint16_t)};
const HI_DataType HI_kInt32 = {Int32, sizeof(int32_t)};
const HI_DataType HI_kUint32 = {Uint32, sizeof(uint32_t)};
const HI_DataType HI_kInt64 = {Int64, sizeof(int64_t)};
const HI_DataType HI_kUint64 = {Uint64, sizeof(uint64_t)};
const HI_DataType HI_kFloat = {Float, sizeof(float)};
const HI_DataType HI_kDouble = {Double, sizeof(double)};
