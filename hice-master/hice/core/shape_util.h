#pragma once

#include <initializer_list>
#include <string>

#include "hice/util/types.h"
#include "hice/core/macros.h"
#include "hice/core/layout_util.h"
#include "hice/core/util.h"
#include "hice/core/shape.h"

namespace hice {

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class HICE_API ShapeUtil {
 public:
  // Returns whether the lhs and rhs shapes are identical.
  static bool equal(const Shape& lhs, const Shape& rhs);

  // Returns whether the LHS and RHS shapes have the same dimensions; note: does
  // not check element type.
  // Precondition: IsArray(lhs) && IsArray(rhs)
  static bool same_dimensions(const Shape& lhs, const Shape& rhs);

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool compatible(const Shape& lhs, const Shape& rhs);

  // Returns true if `shape` (which must be an array) with degenerate dimensions
  // (dimensions with bound 1).
  static bool has_degenerate_dimensions(const Shape& shape);

  // Drops any degenerate dimensions (i.e. dimensions of size 1)
  static Shape delete_degenerate_dimensions(const Shape& shape);

  // Returns a shape with the given dimension deleted.
  // For example:
  // • `delete_dimension(1, T[m, n, k]) = T[m, k]`
  static Shape delete_dimension(int64_t dim_to_delete, Shape shape);

  // Returns a shape with all the dimensions of the input shape for which `p`
  // returns true.
  // For examples:
  // • `filter_dimensions((< 2), T[m, n, k]) = T[m, n]`
  // • `filter_dimensions(is_even_number, T[m, n, k]) = T[m, k]`
  static Shape filter_dimensions(const std::function<bool(int64_t)>& p,
                                 Shape shape);

  // Returns the number of elements are contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  static int64_t get_num_items(const Shape& shape);

  // Returns true if 'shape' is for a scalar
  static bool is_scalar(const Shape& shape);

  // Returns true if 'shape' is an array with zero elements. This means at least
  // one element of the shape's dimensions should be zero
  static bool is_zero_item(const Shape& shape);

  // Returns the number of dimensions for which the dimension is not
  // (trivially)
  // 1. e.g., f32[2x1x1] has a true rank of 1D, the other dimensions are just
  // fluff. Note that zero dimensions are included in the true rank, e.g.,
  // f32[3,0,1] has a true rank of 2D.
  static int64_t get_true_rank(const Shape& shape);

  // Extracts the size of the shape's dimension at dimension number
  // get_dimension_number(dimension_number).
  static int64_t get_dimension(const Shape& shape, int64_t dimension_number);

  // Resolves a dimension number, supporting negative indexing.
  //
  // Negative indexing has similar semantics to Python. For an N-dimensional
  // array, dimension -1 is equivalent to dimension N-1, -2 is equivalent to
  // N-2, and so on.
  //
  // This function always returns a positive dimension number for any given
  // dimension_number (which itself can be negative).
  static int64_t get_dimension_number(const Shape& shape,
                                      int64_t dimension_number);

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static std::string human_string(const Shape& shape);
  static std::string human_string_with_layout(const Shape& shape);

  // Appends a major dimension to the shape with the given bound.
  static void append_major_dimension(int bound, Shape& shape);

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static Shape make_shape(ConstIntArrayRef dimensions);

  // As make_shape, but the object to write to is passed in.
  static bool populate_shape(ConstIntArrayRef dimensions,
                             Shape& shape);

  // Constructs a new shape with the given minor_to_major order in its Layout.
  // Returns a value shape such that shape.has_layout().
  static Shape make_shape_with_layout(ConstIntArrayRef dimensions,
                                      ConstIntArrayRef minor_to_major);

  // Constructs a new shape with major-first layout (i.e. {n, n-1, ..., 0}).
  static Shape make_shape_with_descending_layout(
      ConstIntArrayRef dimensions);

  // Returns a new Shape based on the given Shape with low-dimension-major
  // layout (i.e. {n, n-1, ..., 0}, like Fortran), and with the dimensions
  // rearranged so that it has the same in-memory layout as the given shape.
  //
  // For example, transforms f32[B,H,W,C]{0,3,2,1} to f32[H,W,C,B]{3,2,1,0}.
  static Shape make_shape_with_descending_layout_and_same_physical_layout(
      const Shape& shape);

  // Validates that the provided shape satisfies invariants.
  static bool validate_shape(const Shape& shape);

  // Validates the provided shape satisfies invariants, except those that
  // pertain to layout.
  //
  // Layout is optional for client-provided shapes, so that the compiler may
  // determine and assign an optimized layout.
  static bool validate_shape_with_optional_layout(const Shape& shape);

  // Permutes the dimensions by the given permutation, so
  // return_value.dimensions[permutation[i]] = argument.dimensions[i].
  //
  // Postcondition: For any valid permutation,
  //
  //   !HasLayout(shape) ||
  //   transpose_is_bitcast(shape, permute_dimensions(permutation, shape),
  //                      inverse_permutation(permutation)).
  static Shape permute_dimensions(ConstIntArrayRef permutation,
                                  const Shape& shape);

  // If we can go from `shape_pre` to `shape_post` by merely inserting or
  // deleting 1-sized dimensions, return the indices in `shape_pre` of the
  // deleted dimensions and the indices in `dims_post` of the inserted
  // dimensions.
  // For example, if `shape_pre = {a_1, a_2, ..., a_m}` and
  // `shape_post = {b_1, b_2, ..., b_n}` where we can find some sequence of
  // `i`s and some sequence of `j`s so `a_i = 1` for each `i` and `b_j = 1`
  // for each `j` and `a_(k-s) = b_(k-t)` where `s` and `t` are the number of
  // `i`s and `j`s less than `k` for all other `k`, we return the `i`s and
  // `j`s. For another example, if `shape_pre = shape_post = {}`, we return
  // `{}`.
  static std::tuple<bool, std::vector<int64_t>, std::vector<int64_t>>
  inserted_or_deleted_1sized_dimensions(const Shape& shape_pre,
                                        const Shape& shape_post);

  // Suppose a reshape transforms input_shape to output shape. Returns a
  // vector of pairs that indicate the input and output dimensions that this
  // reshape doesn't logically (i.e. ignoring the layout) modify. For each
  // pair (I,O) in the returned vector, the reshape transforms any input index
  // whose I-th dimension is x to an output index whose O-th dimension is x
  // too.
  //
  // Post-condition: the returned vector is sorted (by both input and output
  // dimensions because input and output dimensions have the same order).
  //
  // Example:
  //   input  shape = T[a, b, x, y, cd]
  //   output shape = T[ab, x, 1, y, c, d]
  //   return value = {{2, 1}, {3, 3}}
  //
  //   The two pairs represent the input and output dimension of size x and
  //   those of size y.
  static std::vector<std::pair<int64_t, int64_t>>
  dimensions_unmodified_by_reshape(const Shape& input_shape,
                                   const Shape& output_shape);

  // Returns whether a transpose from input_shape to output_shape with
  // dimension mapping "dimension_mapping" produces a result which is bit-wise
  // identical to its input and thus may be replaced with a bitcast.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool transpose_is_bitcast(const Shape& input_shape,
                                   const Shape& output_shape,
                                   ConstIntArrayRef dimension_mapping);

  // Returns whether a reshape from "input_shape" to "output_shape" is a
  // bitcast.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool reshape_is_bitcast(const Shape& input_shape,
                                 const Shape& output_shape);

  // Find a physical layout for 'output_shape' such that
  // ShapeUtil::reshape_is_bitcast(input_shape, output_shape_with_layout)
  // returns true (where 'output_shape_with_layout' is 'output_shape' with the
  // found layout). The layout of 'input_shape' is kept fixed. Returns
  // 'output_shape_with_layout' if such a layout can be found, and an error
  // otherwise.
  static optional<Shape> align_layouts(const Shape& input_shape,
                                             const Shape& output_shape);

 private:
  // Validates the shape size is sane. This makes sure it's safe to do
  // calculations in int64_t without overflowing.
  static bool validate_shape_size(const Shape& shape);

  // Validates all of the non-layout properties of the shape -- this is a
  // helper used by both the layout-optional and layout-required public
  // method.
  static bool validate_shape_size_with_optional_layout_internal(
      const Shape& shape);

  HICE_DISABLE_COPY_AND_ASSIGN(ShapeUtil);
};

}  // namespace hice