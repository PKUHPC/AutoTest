#pragma once

#include "hice/api_c/tensor.h"
#include "hice/core/tensor.h"

typedef struct HI_Tensor_Impl {
  hice::Tensor tensor_;
} HI_Tensor_Impl;