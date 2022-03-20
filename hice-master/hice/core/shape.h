#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "hice/core/layout.h"

namespace hice {

// A shape describes the number of dimensions in a array, the bounds of
// each dimensions, and the primitive component type. For tuples, shape
// describes the structure (number of elements and nesting).
class HICE_API Shape {
 public:
  Shape() = default;
  //explicit Shape(ConstIntArrayRef dimensions);
  Shape(ConstIntArrayRef dimensions, const Layout &layout);

  int64_t rank() const { return dimensions_.size(); }

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  std::string to_string(bool print_layout = false) const;

  // Methods for accessing the dimensions array.
  int64_t dimensions_size() const { return dimensions_.size(); }
  int64_t dimensions(int index) const { return dimensions_.at(index); }
  void set_dimensions(int index, int64_t value) {
    dimensions_.at(index) = value;
  }
  // Actually replace the old dimensions with the new dimensions
  void set_dimensions(ConstIntArrayRef dimensions) {
    dimensions_.resize(dimensions.size());
    std::copy(dimensions.begin(), dimensions.end(), dimensions_.begin());
  }
  void add_dimensions(int64_t value) { dimensions_.push_back(value); }
  // Removes the given dimension form the shape. Layout, if it exists, is
  // adjusted to match the modified shape.
  void delete_dimension(int64_t dimension_to_delete);
  void clear_dimensions() { dimensions_.clear(); }
  const std::vector<int64_t>& dimensions() const { return dimensions_; }
  std::vector<int64_t>& mutable_dimensions() { return dimensions_; }

  // Methods for accessing the layout field.
  bool has_layout() const { return layout_.type() != LayoutType::Invalid; }
  const Layout& layout() const { return layout_; }
  Layout& mutable_layout() { return layout_; }
  void clear_layout() { layout_.clear(); }

  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const { return !(*this == other); }

  void swap(Shape &other) { std::swap(*this, other); }

  void clear() {
    clear_dimensions();
    clear_layout();
  }

 private:
  std::vector<int64_t> dimensions_;
  Layout layout_;
};

HICE_API std::ostream& operator<<(std::ostream& out, const Shape& shape);

HICE_API std::ostream& operator<<(std::ostream& out, ConstIntArrayRef arr);

HICE_API std::ostream& operator<<(std::ostream& out, std::vector<int64_t> arr);

}  // namespace hice 
