#pragma once

#include <string>

#include "hice/util/types.h"
#include "hice/core/layout.h"
#include "hice/core/layout_util.h"
#include "hice/core/macros.h"
#include "hice/core/shape.h"

namespace hice {

// Namespaced collection of (static) Layout utilities.
class HICE_API LayoutUtil {
 public:
  // Returns whether the given Layout has a dense layout.
  static bool is_dense(const Layout& layout);

  // Returns whether the given Shape has a dense layout
  static bool is_dense(const Shape& shape);

  // Returns whether the given Shape has a dense layout
  static bool is_default_layout(const Shape& shape);

  // Returns whether the given Layout has a coo layout.
  static bool is_coo(const Layout& layout);

  // Returns whether the given shape has a layout. For tuple shapes, true is
  // returned only if all elements have layouts.
  static bool has_layout(const Shape& shape);

  // Returns whether lhs and rhs are identical.
  static bool equal(const Layout& lhs, const Layout& rhs);

  // Clears the layout in the given Shape. After this function is called,
  // has_layout will return false for the shape.
  static void clear_layout(Shape& shape);

  // Returns whether the given dimensions are consecutive in the given layout,
  // not necessarily in the order given.
  static bool are_dims_consecutive(const Layout& layout,
                                   ConstIntArrayRef dims);

  // Returns default layout for the given shape.
  static Layout get_default_layout_for_shape(const Shape& shape);

  // Helper functions that create default layouts for various ranks.
  static Layout get_default_layout_for_rank(int64_t rank);

  // Returns a shape with the same dimensions as `shape` but with the default
  // layout.
  static Shape get_shape_with_default_layout(const Shape& shape);

  // Sets the default layout on the Shape.
  static void set_to_default_layout(Shape& shape);

  // Sets the default minor_to_major on Layout.
  static void set_to_default_minor_to_major(Layout& layout);

  // Creates a layout with the given minor-to-major dimension order. (This is a
  // convenience function for protobuf construction.)
  static Layout make_layout(ConstIntArrayRef minor_to_major);

  // Returns a layout with descending ((i.e. {n, n-1, ..., 0}) minor-to-major
  // dimensions.
  static Layout make_descending_layout(int64_t rank);

  // Similar to make_layout, but take indices in reverse order.
  static Layout make_layout_from_major_to_Minor(
      ConstIntArrayRef major_to_minor);

  // Validates that the layout within the given shape is correct. The check
  // is performed for all subshapes as well. If missing layouts are allowed
  // the check does not fail on array shapes without layouts.
  static bool validate_layout_in_shape(const Shape& shape,
                                       bool allow_missing_layouts = false);

  // Validates that the provided layout satisfies invariants for the given
  // shape.
  static bool validate_layout_for_shape(const Layout& layout,
                                        const Shape& shape);

  // Returns whether the layout is monotonic and dim 0 is minor in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to column-major. Dimension 0 is the minor, dimension 1 is
  //        more major, and so on until dimension N-1 which is the major.
  static bool is_monotonic_with_dim0_Minor(const Layout& layout);

  // Returns whether the layout is monotonic and dim 0 is major in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to row-major. Dimension 0 is the major, dimension 1 is
  //        more minor, and so on until dimension N-1 which is the minor.
  static bool is_monotonic_with_dim0_major(const Layout& layout);

  // Returns the minor_to_major array for the given Shape.  Requires that the
  // shape is an array and has a dense layout.
  static ConstIntArrayRef minor_to_major(const Shape& shape);
  static ConstIntArrayRef minor_to_major(const Layout& layout);

  // Major(0) is the most major logical dimension number, Major(1) is the
  // second-most-major logical dimension number and so on.
  //
  // This can be used to translate physical dimension numbers to logical
  // dimension numbers. Assume that we are numbering the physical dimensions so
  // that the most major physical dimension has physical dimension number 0 and
  // so on. Then a physical dimension number p corresponds to the logical
  // dimension number Major(p). So this function could also be called
  // PhysicalToLogical().
  //
  // As an example, consider physical dimension number 0, which by definition is
  // the most major. Then Major(0) is the most major logical dimension, so Major
  // maps the physical dimension number 0 to the most major logical dimension
  // number Major(0).
  static int64_t Major(const Layout& layout, int64_t physical_dimension_number);

  // Minor(0) is the most minor logical dimension number, Minor(1) is the
  // second-most-minor logical dimension number and so on.
  static int64_t Minor(const Layout& layout, int64_t physical_dimension_number);

  // Returns the inverse mapping of the Major() function. More precisely, return
  // a vector v such that if l == Major(p), then v[l] == p.
  //
  // This can be used to translate logical dimension numbers into physical
  // dimension numbers. Assume that we are numbering the physical dimensions so
  // that the most major physical dimension has physical dimension number 0 and
  // so on. Then a logical dimension number l corresponds to the physical
  // dimension number make_logical_to_physical(layout)[l].
  //
  // As an example, consider physical dimension number 0, which by definition is
  // the most major. Then l := Major(0) is the most major logical dimension. If
  // v is the vector returned from this function, then v[l] == 0. So v maps the
  // most major logical dimension l to the physical dimension number 0.
  static std::vector<int64_t> make_logical_to_physical(const Layout& layout);

  // Copies the layout from 'src' to 'dst'. Recursively copies layouts of
  // tuples.  'src' and 'dst' need not be compatible but the two shapes must
  // have the same tuple structure (if any) and arrays must have the same
  // rank. within the shapes must have the same number of dimensions.
  static bool copy_layout_between_shapes(const Shape& src, Shape& dst);

  // Returns true if the layouts of lhs and rhs are equal, false
  // otherwise. Recursively compares layouts of tuples.
  //
  // lhs and rhs need not be compatible to have the same layout but the two
  // shapes must have the same tuple structure (if any) and arrays must have the
  // same rank. Element type is ignored.
  static bool layouts_in_shapes_equal(const Shape& lhs, const Shape& rhs);

  // Returns a human-readable string that represents the given layout.
  static std::string human_string(const Layout& layout);

 private:
   HICE_DISABLE_COPY_AND_ASSIGN(LayoutUtil);
};

}  // namespace hice
