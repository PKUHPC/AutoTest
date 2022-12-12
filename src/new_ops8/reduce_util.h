#ifndef REDUCE_UTIL
#define REDUCE_UTIL

#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/basic/index_utils.h"
#include "src/basic/squeeze.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

static Status permute_dims(const int64_t* stride, const int64_t stride_length,
                           const int64_t* perm, int64_t* permuted_strides) {
  for (int i = 0; i < stride_length; i++) {
    permuted_strides[i] = stride[perm[i]];
  }
}

static Status should_swap(const int64_t* strides, int64_t dim0, int64_t dim1) {
  return strides[dim0] < strides[dim1];
}

static Status int64_swap(int64_t* a, int64_t* b) {
  int64_t tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
}

static Status reorder_dims(int64_t* strides, int64_t strides_length,
                           int64_t* perm) {
  for (int i = 0; i < strides_length; i++) {
    perm[i] = i;
  }
  for (int i = 1; i < strides_length; ++i) {
    for (int j = i; j > 0; j--) {
      int comparsion = should_swap(strides, perm[j], perm[j - 1]);
      if (comparsion) {
        int64_swap(perm + j, perm + j - 1);
      } else {
        break;
      }
    }
  }
}

static Status strides_for_computing(const int64_t* strides_old,
                                    const int64_t* dims, const int64_t ndim,
                                    int64_t length_stride,
                                    int64_t* strides_new) {
  int64_t offset = length_stride - ndim;
  memset(strides_new, 0, sizeof(strides_new));
  for (int j = 0; j < ndim; ++j) {
    if (dims[j] != 1) {
      strides_new[j + offset] = strides_old[j];
    }
  }
}

static int64_t int64_max(int64_t x, int64_t y) {
  if (x > y)
    return x;
  else
    return y;
}

#endif  // REDUCE_UTIL