#pragma once

#include "mkl.h"
#include "mkl_spblas.h"

namespace hice {

inline const char* mklsparseGetErrorString(sparse_status_t error) {
  switch (error) {
  case SPARSE_STATUS_SUCCESS:
    return "SPARSE_STATUS_SUCCESS";
  case SPARSE_STATUS_NOT_INITIALIZED:
    return "SPARSE_STATUS_NOT_INITIALIZED";
  case SPARSE_STATUS_ALLOC_FAILED:
    return "SPARSE_STATUS_ALLOC_FAILED";
  case SPARSE_STATUS_INVALID_VALUE:
    return "SPARSE_STATUS_INVALID_VALUE";
  case SPARSE_STATUS_EXECUTION_FAILED:
    return "SPARSE_STATUS_EXECUTION_FAILED";
  case SPARSE_STATUS_INTERNAL_ERROR:
    return "SPARSE_STATUS_INTERNAL_ERROR";
  case SPARSE_STATUS_NOT_SUPPORTED:
    return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  return "Unrecognized mklsparse error string";
}

#define HICE_MKLSPARSE_CHECK(condition) {         \
  sparse_status_t status = (condition);           \
  HICE_CHECK(status == SPARSE_STATUS_SUCCESS)     \
      << mklsparseGetErrorString(status);         \
}

} // namespace hice