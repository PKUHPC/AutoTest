#include "hice/api_c/status.h"
#include "hice/api_c/error_handle.h"

const char* HI_GetLastError() { return hice::GetLastError(); }

void HI_CheckStatus(HI_Status status) {
  if (status != HI_Status::Success) {
    fprintf(stderr, "%s%s%s%s\n", loguru::terminal_reset(),
            loguru::terminal_red(), hice::GetLastError(),
            loguru::terminal_reset());
  }
}

HI_Status Str2Status(const char* c_str) {
  std::string str(c_str);
  if (str.compare("Success")) {
     return Success; 
  } else if (str.compare("TypeMismatch")) {
     return TypeMismatch; 
  } else if (str.compare("DimensionsMismatch")) {
     return DimensionsMismatch; 
  } else if (str.compare("InvalidArgument")) {
     return InvalidArgument; 
  } else if (str.compare("NotSupported")) {
     return NotSupported; 
  } else if (str.compare("AllocFailed")) {
     return AllocFailed; 
  } else if (str.compare("OutOfRange")) {
     return OutOfRange; 
  } else if (str.compare("InternalError")) {
     return InternalError; 
  } else if (str.compare("MathError")) {
     return MathError; 
  } else {
     return UnknownError;
  }
}

const char* Status2Str(HI_Status status) {
  if (status == Success) {
    return "Success";
  } else if (status == TypeMismatch) {
    return "TypeMismatch";
  } else if (status == DimensionsMismatch) {
    return "DimensionsMismatch";
  } else if (status == InvalidArgument) {
    return "InvalidArgument";
  } else if (status == NotSupported) {
    return "NotSupported";
  } else if (status == AllocFailed) {
    return "AllocFailed";
  } else if (status == OutOfRange) {
    return "OutOfRange";
  } else if (status == InternalError) {
    return "InternalError";
  } else if (status == MathError) {
    return "MathError";
  } else {
    return "UnknownError";
  }
}