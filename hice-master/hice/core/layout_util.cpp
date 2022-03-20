#include <stddef.h>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "hice/util/types.h"
#include "hice/util/loguru.h"
#include "hice/core/util.h"
#include "hice/core/layout_util.h"

namespace hice {

bool LayoutUtil::is_dense(const Layout& layout) {
  return layout.type() == kDense;
}

bool LayoutUtil::is_dense(const Shape& shape) {
  return shape.has_layout() && is_dense(shape.layout());
}

bool LayoutUtil::is_default_layout(const Shape& shape) {
  auto& minor_to_major = shape.layout().minor_to_major();
  const int64_t size = minor_to_major.size();
  for (int64_t i = 0; i < size; ++i) {
    if (minor_to_major[i] != size - 1 - i) return false;
  }
  return true;
}

bool LayoutUtil::is_coo(const Layout& layout) {
  return layout.type() == kCOO;
}

bool LayoutUtil::equal(const Layout& lhs, const Layout& rhs) {
  return lhs == rhs;
}

bool LayoutUtil::layouts_in_shapes_equal(const Shape& lhs, const Shape& rhs) {
  return (lhs.rank() == rhs.rank()) &&
         (LayoutUtil::equal(lhs.layout(), rhs.layout()));
}

bool LayoutUtil::has_layout(const Shape& shape) { return shape.has_layout(); }

void LayoutUtil::clear_layout(Shape &shape) { shape.clear_layout(); }

bool LayoutUtil::are_dims_consecutive(const Layout& layout,
                                      ConstIntArrayRef dims) {
  std::vector<int64_t> positions_in_layout;
  for (int64_t dim : dims) {
    positions_in_layout.push_back(
        position_in_container(layout.minor_to_major(), dim));
  }
  // Why this is sorted
  c_sort(positions_in_layout);
  for (size_t i = 1; i < positions_in_layout.size(); ++i) {
    if (1 != positions_in_layout[i] - positions_in_layout[i - 1]) {
      return false;
    }
  }
  return true;
}

namespace {

// Internal helper for get_default_layout_from_shape and set_to_default_layout.
// Sets minor_to_major to the value that represents the default layout.
void set_default_layout_to_container(std::vector<int64_t>& minor_to_major) {
  // The default hice layout is major-to-minor (dim 0 is major).
  const int64_t size = minor_to_major.size();
  for (int64_t i = 0; i < size; ++i) {
    minor_to_major[i] = size - 1 - i;
  }
}

// Internal helper that creates a default layout for an array of the given rank.
Layout create_default_layout_for_rank(int64_t rank) {
  Layout layout;
  layout.set_type(kDense);
  std::vector<int64_t>& minor_to_major = layout.mutable_minor_to_major();
  minor_to_major.resize(rank, 0);
  set_default_layout_to_container(minor_to_major);
  return layout;
}

}  // namespace

Layout LayoutUtil::get_default_layout_for_shape(const Shape& shape) {
  return create_default_layout_for_rank(shape.rank());
}

Layout LayoutUtil::get_default_layout_for_rank(int64_t rank) {
  return create_default_layout_for_rank(rank);
}

void LayoutUtil::set_to_default_layout(Shape& shape) {
  shape.mutable_layout().set_type(kDense);
  auto& minor_to_major = shape.mutable_layout().mutable_minor_to_major();
  minor_to_major.resize(shape.rank(), 0);
  set_default_layout_to_container(minor_to_major);
}

Shape LayoutUtil::get_shape_with_default_layout(const Shape& shape) {
  Shape copy(shape);
  LayoutUtil::set_to_default_layout(copy);
  return copy;
}

Layout LayoutUtil::make_layout(ConstIntArrayRef minor_to_major) {
  Layout layout;
  layout.set_type(kDense);
  for (int64_t dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
  }
  return layout;
}

Layout LayoutUtil::make_descending_layout(int64_t rank) {
  std::vector<int64_t> layout(rank);
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return make_layout(layout);
}

Layout LayoutUtil::make_layout_from_major_to_Minor(
    ConstIntArrayRef major_to_minor) {
  Layout layout;
  layout.set_type(kDense);
  for (int i = major_to_minor.size() - 1; i >= 0; i--) {
    layout.add_minor_to_major(major_to_minor[i]);
  }
  return layout;
}

bool LayoutUtil::validate_layout_in_shape(const Shape& shape,
                                          bool allow_missing_layouts) {
  if (!shape.has_layout()) {
    if (allow_missing_layouts) {
      return true;
    }
    return false;
  }
  return validate_layout_for_shape(shape.layout(), shape);
}

bool LayoutUtil::validate_layout_for_shape(const Layout& layout,
                                           const Shape& shape) {
  if (layout.type() == LayoutType::Invalid) {
    return false;
  }
  // Right now we just check the layout for dense shape
  if (layout.type() == kDense) {
    if (layout.minor_to_major_size() != shape.rank()) {
      return false;
    }
    std::vector<bool> dimensions_in_layout(shape.rank(), false);
    for (int64_t i = 0; i < shape.rank(); ++i) {
      int64_t dim = layout.minor_to_major(i);
      if (dim < 0 || dim >= shape.rank()) {
        return false; // layout minor_to_major field has out-of-bounds value
      }
      if (dimensions_in_layout[dim]) {
        return false; // layout minor_to_major field has duplicate values
      }
      dimensions_in_layout[dim] = true;
    }
  }
  return true;
}

bool LayoutUtil::is_monotonic_with_dim0_Minor(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

bool LayoutUtil::is_monotonic_with_dim0_major(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64_t>());
}

ConstIntArrayRef LayoutUtil::minor_to_major(const Shape& shape) {
  return as_int64_slice(shape.layout().minor_to_major());
}

ConstIntArrayRef LayoutUtil::minor_to_major(const Layout& layout) {
  return as_int64_slice(layout.minor_to_major());
}

int64_t LayoutUtil::Major(const Layout& layout,
                          int64_t physical_dimension_number) {
  HICE_CHECK_LE(0, physical_dimension_number);
  HICE_CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return Minor(layout,
               layout.minor_to_major_size() - 1 - physical_dimension_number);
}

int64_t LayoutUtil::Minor(const Layout& layout,
                          int64_t physical_dimension_number) {
  HICE_CHECK_LE(0, physical_dimension_number);
  HICE_CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return layout.minor_to_major(physical_dimension_number);
}

std::vector<int64_t> LayoutUtil::make_logical_to_physical(
    const Layout& layout) {
  std::vector<int64_t> logical_to_physical(layout.minor_to_major_size());
  for (int64_t physical = 0; physical < logical_to_physical.size();
       ++physical) {
    const int64_t logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}

std::string LayoutUtil::human_string(const Layout& layout) {
  return layout.to_string();
}

namespace {

// Internal helper for recursively copying layouts.
bool copy_layout_internal(const Shape& src, Shape& dst) {
  if (src.has_layout()) {
    if (src.rank() != dst.rank()) {
      return false; // cannot copy layout if their ranks differs
    }
    if (!LayoutUtil::validate_layout_for_shape(src.layout(), dst)) return false;
    dst.mutable_layout() = src.layout();
  } else {
    dst.clear_layout();
  }
  return true;
}

}  // namespace

bool LayoutUtil::copy_layout_between_shapes(const Shape& src, Shape &dst) {
  return copy_layout_internal(src, dst);
}

}  // namespace hice
