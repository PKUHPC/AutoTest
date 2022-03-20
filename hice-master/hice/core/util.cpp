#include <stdarg.h>
#include <numeric>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"

#include "hice/core/util.h"

namespace hice {

bool is_permutation(ConstIntArrayRef permutation, int64_t rank) {
  if (rank != permutation.size()) {
    return false;
  }
  std::vector<int64_t> output(permutation.size(), -1);
  for (auto index : permutation) {
    HICE_CHECK_GE(index, 0);
    HICE_CHECK_LT(index, rank);
    output[index] = 0;
  }
  return !c_linear_search(output, -1);
}

std::vector<int64_t> inverse_permutation(
    ConstIntArrayRef input_permutation) {
  HICE_DCHECK(is_permutation(input_permutation, input_permutation.size()));
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

std::vector<int64_t> compose_permutations(ConstIntArrayRef p1,
                                          ConstIntArrayRef p2) {
  HICE_CHECK_EQ(p1.size(), p2.size());
  std::vector<int64_t> output;
  for (size_t i = 0; i < p1.size(); ++i) {
    output.push_back(p1[p2[i]]);
  }
  return output;
}

bool is_identity_permutation(ConstIntArrayRef permutation) {
  for (int64_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

int64_t product(ConstIntArrayRef xs) {
  return std::accumulate(xs.begin(), xs.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

std::vector<std::pair<int64_t, int64_t>> common_factors(
    ConstIntArrayRef a, ConstIntArrayRef b) {
  HICE_CHECK_EQ(product(a), product(b));
  if (0 == product(a)) {
    return {std::make_pair(0, 0), std::make_pair(a.size(), b.size())};
  }

  std::vector<std::pair<int64_t, int64_t>> bounds;
  for (int64_t i = 0, j = 0, prior_i = -1, prior_j = -1, partial_size_a = 1,
               partial_size_b = 1;
       ;) {
    if (partial_size_a == partial_size_b && (i > prior_i || j > prior_j)) {
      std::tie(prior_i, prior_j) = std::make_pair(i, j);
      bounds.emplace_back(i, j);
      continue;
    }
    bool in_bounds_i = i < a.size();
    bool in_bounds_j = j < b.size();
    if (!(in_bounds_i || in_bounds_j)) {
      break;
    }
    bool next_a =
        partial_size_a < partial_size_b ||
        (in_bounds_i &&
         (!in_bounds_j || (partial_size_a == partial_size_b && a[i] <= b[j])));
    bool next_b =
        partial_size_b < partial_size_a ||
        (in_bounds_j &&
         (!in_bounds_i || (partial_size_b == partial_size_a && b[j] <= a[i])));
    if (next_a) {
      partial_size_a *= a[i];
      ++i;
    }
    if (next_b) {
      partial_size_b *= b[j];
      ++j;
    }
  }
  return bounds;
}

}  // namespace hice
