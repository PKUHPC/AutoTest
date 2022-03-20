#pragma once

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/utility/utility.h"

namespace hice {

template <typename T>
using span = absl::Span<T>;

template <typename T>
using ArrayRef = span<T>;
using IntArrayRef = span<int64_t>;
template <typename T>
using ConstArrayRef = span<const T>;
using ConstIntArrayRef = span<const int64_t>;

using absl::make_optional;
using absl::nullopt;
using absl::nullopt_t;
using absl::optional;

using absl::in_place;
using absl::in_place_t;

}  // namespace hice