#pragma once
#include "hice/core/export.h"

namespace hice {

enum class Status : int16_t {
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
};

constexpr Status kSuccess = Status::Success;
constexpr Status kTypeMismatch = Status::TypeMismatch;
constexpr Status kDimensionsMismatch = Status::DimensionsMismatch;
constexpr Status kInvalidArgument = Status::InvalidArgument;
constexpr Status kNotSupported = Status::NotSupported;
constexpr Status kAllocFailed = Status::AllocFailed;
constexpr Status kOutOfRange = Status::OutOfRange;
constexpr Status kInternalError = Status::InternalError;
constexpr Status kMathError = Status::MathError;
constexpr Status kUnknownError = Status::UnknownError;

inline Status str2status(std::string str) {
  if (str.compare("Success")) {
     return kSuccess; 
  } else if (str.compare("TypeMismatch")) {
     return kTypeMismatch; 
  } else if (str.compare("DimensionsMismatch")) {
     return kDimensionsMismatch; 
  } else if (str.compare("InvalidArgument")) {
     return kInvalidArgument; 
  } else if (str.compare("NotSupported")) {
     return kNotSupported; 
  } else if (str.compare("AllocFailed")) {
     return kAllocFailed; 
  } else if (str.compare("OutOfRange")) {
     return kOutOfRange; 
  } else if (str.compare("InternalError")) {
     return kInternalError; 
  } else if (str.compare("MathError")) {
     return kMathError; 
  } else {
     return kUnknownError;
  }
}

inline std::string status2str(Status status) {
  if (status == kSuccess) {
    return "Success";
  } else if (status == kTypeMismatch) {
    return "TypeMismatch";
  } else if (status == kDimensionsMismatch) {
    return "DimensionsMismatch";
  } else if (status == kInvalidArgument) {
    return "InvalidArgument";
  } else if (status == kNotSupported) {
    return "NotSupported";
  } else if (status == kAllocFailed) {
    return "AllocFailed";
  } else if (status == kOutOfRange) {
    return "OutOfRange";
  } else if (status == kInternalError) {
    return "InternalError";
  } else if (status == kMathError) {
    return "MathError";
  } else {
    return "UnknownError";
  }
}

}  // namespace hice