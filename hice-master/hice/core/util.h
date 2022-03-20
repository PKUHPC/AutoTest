#pragma once

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "hice/util/loguru.h"
#include "hice/util/types.h"
#include "hice/util/container.h"

namespace hice {

// Ranks greater than 5 are very rare, so use InlinedVector<int64_t, 5> to store
// the bounds and indices. And for the rare cases of ranks greater than 8,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
static constexpr int kInlineRank = 5;
using DimensionVector = InlinedVector<int64_t, kInlineRank>;

static inline ConstIntArrayRef as_int64_slice(
    ConstIntArrayRef slice) {
  return slice;
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using their
// operator==. Like std::equal, but forces size equality.
template <typename Container1T, typename Container2T>
bool containers_equal(const Container1T& c1, const Container2T& c2) {
  return ((c1.size() == c2.size()) &&
          std::equal(std::begin(c1), std::end(c1), std::begin(c2)));
}

template <typename Container1T,
          typename ElementType = typename Container1T::value_type>
bool containers_equal(const Container1T& c1,
                      std::initializer_list<ElementType> il) {
  ConstArrayRef<ElementType> c2{il};
  return containers_equal(c1, c2);
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using the predicate
// p. Like std::equal, but forces size equality.
template <typename Container1T, typename Container2T, class PredicateT>
bool containers_equal(const Container1T& c1, const Container2T& c2,
                      PredicateT p) {
  return ((c1.size() == c2.size()) &&
          std::equal(std::begin(c1), std::end(c1), std::begin(c2), p));
}

// Performs a copy of count values from src to dest, using different strides for
// source and destination. The source starting index is src_base, while the
// destination one is dest_base.
template <typename D, typename S>
void strided_copy(ArrayRef<D> dest, int64_t dest_base, int64_t dest_stride,
                  ConstArrayRef<S> src, int64_t src_base, int64_t src_stride,
                  int64_t count) {
  for (; count > 0; --count, dest_base += dest_stride, src_base += src_stride) {
    dest[dest_base] = static_cast<D>(src[src_base]);
  }
}

// Checks whether permutation is a permutation of the [0, rank) integer range.
HICE_API bool is_permutation(ConstIntArrayRef permutation,
                             int64_t rank);

// Applies `permutation` on `input` and returns the permuted array.
// For each i, output[permutation[i]] = input[i].
//
// Precondition:
// 1. `permutation` is a permutation of 0..permutation.size()-1.
// 2. permutation.size() == input.size().
template <typename Container>
std::vector<typename Container::value_type> permute(
    ConstIntArrayRef permutation, const Container& input) {
  using T = typename Container::value_type;
  ConstArrayRef<T> data(input);
  CHECK(is_permutation(permutation, data.size()));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

// Inverts a permutation, i.e., output_permutation[input_permutation[i]] = i.
HICE_API std::vector<int64_t> inverse_permutation(
    ConstIntArrayRef input_permutation);

// Composes two permutations: output[i] = p1[p2[i]].
HICE_API std::vector<int64_t> compose_permutations(
    ConstIntArrayRef p1, ConstIntArrayRef p2);

// Returns true iff permutation == {0, 1, 2, ...}.
HICE_API bool is_identity_permutation(ConstIntArrayRef permutation);

template <typename Container>
int64_t position_in_container(const Container& container, int64_t value) {
  return std::distance(container.begin(), c_find(container, value));
}

HICE_API int64_t product(ConstIntArrayRef xs);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < common_factors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `common_factors(a, b)[common_factors(a, b).size - 1] = (a.size,
// b.size)`
//
// If the given shapes have non-zero size, returns the bounds of the shortest
// possible such subsequences; else, returns `{(0, 0), (a.size, b.size)}`.
HICE_API std::vector<std::pair<int64_t, int64_t>> common_factors(
    ConstIntArrayRef a, ConstIntArrayRef b);

template <typename C, typename Value>
int64_t find_index(const C& c, Value&& value) {
  auto it = c_find(c, std::forward<Value>(value));
  return std::distance(c.begin(), it);
}

template <typename C, typename Value>
void insert_at(C* c, int64_t index, Value&& value) {
  c->insert(c->begin() + index, std::forward<Value>(value));
}

template <typename C>
void erase_at(C* c, int64_t index) {
  c->erase(c->begin() + index);
}

template <typename T>
std::vector<T> array_slice_to_vector(ConstArrayRef<T> slice) {
  return std::vector<T>(slice.begin(), slice.end());
}

template <typename T, size_t N>
std::vector<T> inlined_vector_to_vector(
    const InlinedVector<T, N>& inlined_vector) {
  return std::vector<T>(inlined_vector.begin(), inlined_vector.end());
}

template <typename T>
bool erase_element_from_vector(std::vector<T>* container, const T& value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  HICE_CHECK(it != container->end());
  container->erase(it);
  return true;
}

// Multiply two nonnegative int64_t's, returning negative for overflow
HICE_API inline int64_t multiply_without_overflow(const int64_t x,
                                                  const int64_t y) {
  // Multiply in uint64_t rather than int64_t since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64_t ux = x;
  const uint64_t uy = y;
  const uint64_t uxy = ux * uy;

  // Check if we overflow uint64_t, using a cheap check if both inputs are small
  if ((ux | uy) >> 32 != 0) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    HICE_CHECK(x >= 0 && y >= 0);

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return -1;
  }

  // Cast back to signed.  Any negative value will signal an error.
  return static_cast<int64_t>(uxy);
}

}  // namespace hice