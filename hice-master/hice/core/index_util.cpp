#include <algorithm>
#include <string>
#include <iostream>

#include "hice/util/types.h"
#include "hice/util/string_ops.h"
#include "hice/util/loguru.h"
#include "hice/core/shape_util.h"
#include "hice/core/index_util.h"

namespace hice {

int64_t IndexUtil::multi_index_to_offset(
    const Shape& shape, ConstIntArrayRef multi_index) {
  HICE_DCHECK_EQ(shape.rank(), multi_index.size());

  for (size_t i = 0; i < multi_index.size(); ++i) {
    HICE_DCHECK_GE(multi_index[i], 0);
    HICE_DCHECK_LT(multi_index[i], shape.dimensions(i))
        << "indexing beyond extent in dimension " << i << ":"
        << "\n\tindex: " << StrJoin(multi_index, ",")
        << "\n\tshape: " << ShapeUtil::human_string(shape);
  }

  // Let the array be sized like so for dimensions i from 0 to n-1:
  //
  //   [D{n-1} x D{n-2} x .. x D{0}]
  //
  // Let the order of the dimensions in the minor_to_major field in
  // Layout be:
  //
  //   L(0), L(1), ... , L(n-1)
  //
  // where L(0) is the most-minor dimension and L(n-1) the most-major. The
  // multidimensional index:
  //
  //   [I{0}, I{1}, ... , I{n-1}]
  //
  // then corresponds to the following linear index:
  //
  // offset =
  //   (((  ... + I{L(2)}) * D{L(1)} + I{L(1)}) * D{L(0)} + I{L(0)}
  //
  // or equivalently:
  //
  // offset =
  //   I{L(n-1)} * (D{L(n-2)} * D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-2)} *             (D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-3)} *                         (D{L(n-4)} *     ....    D{L(0)}) +
  //                                   ...                                   +
  //   I{L(2)} *                                         (D{L(1)} * D{L(0)}) +
  //   I{L(1)} *                                                    D{L(0)}  +
  //   I{L(0)}
  //
  // We compute the linear index value by accumulating the terms above from
  // I{L(0)} up to I{L(n-1)}. Scale accumulates the product term D{L(0}} *
  // D{L(1)} * ...

  // Scale factor holding the growing product of D{L(i)} terms.
  int64_t scale = 1;
  int64_t offset = 0;
  bool first = true;
  for (auto dim : LayoutUtil::minor_to_major(shape)) {
    if (first) {
      // Avoid two multiplies on the first loop iteration
      offset = multi_index[dim];
      scale = shape.dimensions(dim);
      first = false;
    } else {
      offset += scale * multi_index[dim];
      scale *= shape.dimensions(dim);
    }
  }
  return offset;
}

std::vector<int64_t> IndexUtil::offset_to_multi_index(const Shape& shape,
                                                      int64_t offset) {
  HICE_DCHECK_GE(offset, 0);
  HICE_DCHECK_LT(offset, ShapeUtil::get_num_items(shape));

  // The following formula computes each element of the multidimensional index
  // (See comments in multi_index_to_offset for notation):
  //
  // I{L(0)} = offset % D{L(0)}
  // I{L(1)} = (offset / D{L(0)}) % D{L(1)}
  // I{L(2)} = (offset / (D{L(0)} * D{L(1)})) % D{L(2)}
  // ...
  std::vector<int64_t> multi_index(shape.rank());

  // Accumulated product D{L(0)} * D{L(1)} * ...
  int64_t divisor = 1;
  for (auto dim : LayoutUtil::minor_to_major(shape)) {
    multi_index[dim] =
        (offset / divisor) % shape.dimensions(dim);
    divisor *= shape.dimensions(dim);
  }
  return multi_index;
}

void IndexUtil::last_multi_index(const Shape& shape, std::vector<int64_t>& multi_index) {
  ConstIntArrayRef dims = shape.dimensions();
  ConstIntArrayRef mtm = LayoutUtil::minor_to_major(shape);
  int64_t ndim = dims.size();
  for (int64_t i = 0 ; i < ndim; ++i) {
    int64_t dim = mtm[i];
    multi_index[dim]--;
    if (multi_index[dim] >= 0) break;
    multi_index[dim] = dims[dim] - 1;
    // HICE_DLOG_IF(INFO, i == ndim - 1) << "Underflow occurs when making decreasment on multi_index.";
  }
}

void IndexUtil::next_multi_index(const Shape& shape, std::vector<int64_t>& multi_index) {
  ConstIntArrayRef dims = shape.dimensions();
  ConstIntArrayRef mtm = LayoutUtil::minor_to_major(shape);
  int64_t ndim = dims.size();
  for (int64_t i = 0 ; i < ndim; ++i) {
    int64_t dim = mtm[i];
    multi_index[dim]++;
    if (multi_index[dim] < dims[dim]) break;
    multi_index[dim] = 0;
    // HICE_DLOG_IF(INFO, i == ndim - 1) << "Overflow occurs when making increasment on multi_index.";
  }
}

bool IndexUtil::bump_indices(const Shape& shape, IntArrayRef indices) {
  for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
    int64_t limit = shape.dimensions(dimno);
    if (indices[dimno] + 1 < limit) {
      indices[dimno]++;
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

int64_t IndexUtil::get_dim_stride(const Shape& shape, int64_t dimension) {
  int64_t stride = 1;
  for (auto dim : LayoutUtil::minor_to_major(shape)) {
    if (dim == dimension) {
      break;
    }
    stride *= shape.dimensions()[dim];
  }
  return stride;
}

std::vector<int64_t> IndexUtil::get_all_strides(const Shape& shape) {
  std::vector<int64_t> strides;
  strides.resize(shape.rank(), 0);
  int64_t stride = 1;
  for (auto dim : LayoutUtil::minor_to_major(shape)) {
    strides.at(dim) = stride;
    stride *= shape.dimensions().at(dim);
  }
  return strides;
}

bool IndexUtil::index_in_bounds(const Shape& shape,
                                ConstIntArrayRef index) {
  int64_t ndim = shape.rank();
  if (ndim != index.size()) {
    return false;
  }
  for (int64_t d = 0; d < ndim; ++d) {
    if (index[d] >= shape.dimensions(d)) {
      return false;
    }
  }
  return true;
}

int IndexUtil::compare_indices(ConstIntArrayRef lhs,
                               ConstIntArrayRef rhs) {
  int64_t ndim = lhs.size();
  CHECK_EQ(rhs.size(), ndim);
  for (int64_t dim = 0; dim < ndim; ++dim) {
    if (lhs[dim] < rhs[dim]) {
      return -1;
    } else if (lhs[dim] > rhs[dim]) {
      return 1;
    }
  }
  return 0;
}

}  // namespace hice
