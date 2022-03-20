#pragma once

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"

namespace hice {
  
using absl::InlinedVector;

using absl::c_equal;
using absl::c_find;
using absl::c_linear_search;
using absl::c_sort;

}  // namespace hice