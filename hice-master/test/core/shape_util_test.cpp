#include "hice/core/shape.h"
#include "hice/core/shape_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ElementsAre;

TEST(ShapeUtilTest, GetDimensionHelperCanNegativeIndex) {
  Shape matrix = ShapeUtil::make_shape({2, 3});
  EXPECT_EQ(3, ShapeUtil::get_dimension(matrix, -1));
  EXPECT_EQ(2, ShapeUtil::get_dimension(matrix, -2));
}

TEST(ShapeUtilTest, GetDimensionHelperExampleInDocumentationTest) {
  auto shape = ShapeUtil::make_shape({1, 2, 3, 4});
  ASSERT_EQ(4, ShapeUtil::get_dimension(shape, -1));
}

#if 0
TEST(ShapeUtilTest, NegativeIndexOobFails) {
  Shape matrix = ShapeUtil::make_shape({2, 3});
  ASSERT_DEATH(ShapeUtil::get_dimension(matrix, -3), "dimension_number >= 0");
}
#endif

TEST(ShapeUtilTest, Rank1DimensionIndexing) {
  Shape shape = ShapeUtil::make_shape({3});
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank2DimensionIndexing) {
  Shape shape = ShapeUtil::make_shape({3, 2});
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank3DimensionIndexing) {
  Shape shape = ShapeUtil::make_shape({3, 2, 7});
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank4DimensionIndexing) {
  Shape shape = ShapeUtil::make_shape({3, 2, 7, 8});
  ASSERT_EQ(8, shape.dimensions(3));
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, CompatibleIdenticalShapes) {
  Shape shape1 = ShapeUtil::make_shape({3, 2});
  Shape shape2 = ShapeUtil::make_shape({3, 2});
  ASSERT_TRUE(ShapeUtil::compatible(shape1, shape2));
}

TEST(ShapeUtilTest, CompatibleNotIdenticalShapes) {
  Shape shape_1 = ShapeUtil::make_shape({3, 2});
  auto& layout_1 = shape_1.mutable_layout();
  layout_1.clear_minor_to_major();
  layout_1.add_minor_to_major(0);
  layout_1.add_minor_to_major(1);

  Shape shape_2 = ShapeUtil::make_shape({3, 2});
  auto& layout_2 = shape_2.mutable_layout();
  layout_2.clear_minor_to_major();
  layout_2.add_minor_to_major(1);
  layout_2.add_minor_to_major(0);

  EXPECT_FALSE(ShapeUtil::equal(shape_1, shape_2));
  //EXPECT_TRUE(ShapeUtil::compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, ScalarDefaultLayoutEqualsScalarEmptyMin2Maj) {
  Shape scalar_default_layout = ShapeUtil::make_shape({});
  ASSERT_TRUE(scalar_default_layout.has_layout())
      << ShapeUtil::human_string_with_layout(scalar_default_layout);

  const Shape scalar_empty_min2maj = ShapeUtil::make_shape_with_layout({}, {});
  ASSERT_TRUE(scalar_empty_min2maj.has_layout())
      << ShapeUtil::human_string_with_layout(scalar_empty_min2maj);

  EXPECT_TRUE(ShapeUtil::equal(scalar_default_layout, scalar_empty_min2maj));
}

TEST(ShapeUtilTest, GetNumItems) {
  EXPECT_EQ(1, ShapeUtil::get_num_items(ShapeUtil::make_shape({})));
  EXPECT_EQ(0, ShapeUtil::get_num_items(ShapeUtil::make_shape({0})));
  EXPECT_EQ(1, ShapeUtil::get_num_items(ShapeUtil::make_shape({1})));
  EXPECT_EQ(1, ShapeUtil::get_num_items(ShapeUtil::make_shape({1, 1})));
  EXPECT_EQ(2, ShapeUtil::get_num_items(ShapeUtil::make_shape({2})));
  EXPECT_EQ(2, ShapeUtil::get_num_items(ShapeUtil::make_shape({2, 1})));
  EXPECT_EQ(15, ShapeUtil::get_num_items(ShapeUtil::make_shape({3, 5})));
  EXPECT_EQ(0, ShapeUtil::get_num_items(ShapeUtil::make_shape({3, 0, 5})));
  EXPECT_EQ(0, ShapeUtil::get_num_items(ShapeUtil::make_shape({0, 3, 0})));
  EXPECT_EQ(15, ShapeUtil::get_num_items(ShapeUtil::make_shape({1, 3, 5})));
  EXPECT_EQ(221, ShapeUtil::get_num_items(ShapeUtil::make_shape({13, 17})));
}

TEST(ShapeUtilTest, IsZeroItem) {
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({})));
  EXPECT_TRUE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({0})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({1})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({1, 1})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({2})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({2, 1})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({3, 5})));
  EXPECT_TRUE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({3, 0, 5})));
  EXPECT_TRUE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({0, 3, 0})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({1, 3, 5})));
  EXPECT_FALSE(ShapeUtil::is_zero_item(ShapeUtil::make_shape({13, 17})));
}

TEST(ShapeUtilTest, SameDimensions) {
  EXPECT_TRUE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({}),
                                        ShapeUtil::make_shape({})));
  EXPECT_TRUE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({}),
                                        ShapeUtil::make_shape({})));
  EXPECT_TRUE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1}),
                                        ShapeUtil::make_shape({1})));
  EXPECT_TRUE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({0}),
                                        ShapeUtil::make_shape({0})));
  EXPECT_TRUE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({2}),
                                        ShapeUtil::make_shape({2})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1}),
                                         ShapeUtil::make_shape({2})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({0, 0}),
                                         ShapeUtil::make_shape({0})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1}),
                                         ShapeUtil::make_shape({1, 1})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({}),
                                         ShapeUtil::make_shape({1})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1}),
                                         ShapeUtil::make_shape({1, 1})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1}),
                                         ShapeUtil::make_shape({1, 0})));
  EXPECT_FALSE(ShapeUtil::same_dimensions(ShapeUtil::make_shape({1, 1}),
                                         ShapeUtil::make_shape({1, 2})));
}

TEST(ShapeUtilTest, InsertedOrDeleted1SizedDimensions) {
  Shape shape0 = ShapeUtil::make_shape({9, 1, 4});
  Shape shape1 = ShapeUtil::make_shape({1, 9, 4, 1});
  Shape shape2 = ShapeUtil::make_shape({3, 1, 12});
  EXPECT_TRUE(std::get<0>(
      ShapeUtil::inserted_or_deleted_1sized_dimensions(shape0, shape1)));
  EXPECT_FALSE(std::get<0>(
      ShapeUtil::inserted_or_deleted_1sized_dimensions(shape0, shape2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1x1_to_1x1x1) {
  // All output dimensions should be unmodified. One of the input dimensions is
  // modified because the input rank is larger by one.
  EXPECT_THAT(ShapeUtil::dimensions_unmodified_by_reshape(
                  ShapeUtil::make_shape({1, 1, 1, 1}),
                  ShapeUtil::make_shape({1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1_to_1x1x1x1) {
  // All input dimensions should be unmodified. One of the output dimensions is
  // modified because the output rank is larger by one.
  EXPECT_THAT(ShapeUtil::dimensions_unmodified_by_reshape(
                  ShapeUtil::make_shape({1, 1, 1}),
                  ShapeUtil::make_shape({1, 1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_4x1x3x5x6x7_to_2x6x1x5x1x42) {
  // The only matching dimension is the one with size 5.
  // 4, 1, 3, 5, 6, 7
  //          |
  // 2, 6, 1, 5, 1, 42
  EXPECT_THAT(ShapeUtil::dimensions_unmodified_by_reshape(
                  ShapeUtil::make_shape({4, 1, 3, 5, 6, 7}),
                  ShapeUtil::make_shape({2, 6, 1, 5, 1, 42})),
              ElementsAre(std::make_pair(3, 3)));
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x4_6x2) {
  for (bool input_is_row_major : {true, false}) {
    for (bool output_is_row_major : {true, false}) {
      Layout input_layout = input_is_row_major
                                ? LayoutUtil::make_layout({1, 0})
                                : LayoutUtil::make_layout({0, 1});
      Layout output_layout = output_is_row_major
                                 ? LayoutUtil::make_layout({1, 0})
                                 : LayoutUtil::make_layout({0, 1});
      // Suppose the input is logically (i.e. ignoring its layout)
      //   0  1  2  3
      //   4  5  6  7
      //   8  9  10 11
      //
      // The reshape transforms the input to logically
      //   0  1
      //   2  3
      //   4  5
      //   6  7
      //   8  9
      //   10 11
      //
      // The input and the output have the same underlying data only if they
      // are both row-major.
      EXPECT_EQ(
          ShapeUtil::reshape_is_bitcast(
              ShapeUtil::make_shape_with_layout(
                  {3, 4}, as_int64_slice(input_layout.minor_to_major())),
              ShapeUtil::make_shape_with_layout(
                  {6, 2}, as_int64_slice(output_layout.minor_to_major()))),
          input_is_row_major && output_is_row_major);
    }
  }
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x2x2_6x2_Dim1IsMostMinor) {
  EXPECT_TRUE(ShapeUtil::reshape_is_bitcast(
      ShapeUtil::make_shape_with_layout({3, 2, 2}, {1, 0, 2}),
      ShapeUtil::make_shape_with_layout({6, 2}, {0, 1})));
}

TEST(ShapeUtilTest, HasDegenerateDimensions) {
  EXPECT_TRUE(
      ShapeUtil::has_degenerate_dimensions(ShapeUtil::make_shape({3, 1, 2})));
  EXPECT_TRUE(
      ShapeUtil::has_degenerate_dimensions(ShapeUtil::make_shape({3, 1, 1})));
  EXPECT_FALSE(
      ShapeUtil::has_degenerate_dimensions(ShapeUtil::make_shape({3, 3, 5})));
  EXPECT_FALSE(
      ShapeUtil::has_degenerate_dimensions(ShapeUtil::make_shape({3, 0, 5})));
}

TEST(ShapeUtilTest, PermuteDimensionsLayout) {
  std::vector<int64_t> layout(3);
  std::iota(layout.begin(), layout.end(), 0);
  do {
    Shape s = ShapeUtil::make_shape_with_layout({10, 100, 1000}, layout);
    std::vector<int64_t> permutation(3);
    std::iota(permutation.begin(), permutation.end(), 0);
    do {
      // TransposeIsBitcast takes the inverse of the permutation that
      // PermuteDimensions takes.
      EXPECT_TRUE(ShapeUtil::transpose_is_bitcast(
          s, ShapeUtil::permute_dimensions(permutation, s),
          inverse_permutation(permutation)));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  } while (std::next_permutation(layout.begin(), layout.end()));
}

TEST(AlgebraicSimplifierTest, ReshapeIsBitcast_3x2x2_6x2_Dim0IsMostMinor) {
  EXPECT_FALSE(ShapeUtil::reshape_is_bitcast(
      ShapeUtil::make_shape_with_layout({3, 2, 2}, {0, 1, 2}),
      ShapeUtil::make_shape_with_layout({6, 2}, {0, 1})));
}

TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensions) {
  Shape input = ShapeUtil::make_shape_with_layout({3, 8, 5, 7, 11},
                                                  {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::align_layouts(
      input, ShapeUtil::make_shape({4, 3, 2, 7, 5, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0, 5));
  EXPECT_TRUE(ShapeUtil::reshape_is_bitcast(input, aligned_shape.value()));

  aligned_shape = ShapeUtil::align_layouts(
      input, ShapeUtil::make_shape({3, 2, 4, 35, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(3, 2, 1, 0, 4));
  EXPECT_TRUE(ShapeUtil::reshape_is_bitcast(input, aligned_shape.value()));
}

TEST(AlignmentTest, AlignLayoutsWithTrivialDimensions) {
  Shape input = ShapeUtil::make_shape_with_layout(
      {1, 3, 8, 1, 5, 7, 1, 11, 1, 1}, {5, 0, 4, 2, 1, 3, 6, 7, 9, 8});
  auto aligned_shape = ShapeUtil::align_layouts(
      input, ShapeUtil::make_shape({1, 4, 1, 3, 2, 7, 5, 11, 1}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(6, 5, 4, 3, 1, 7, 0, 2, 8));
  EXPECT_TRUE(ShapeUtil::reshape_is_bitcast(input, aligned_shape.value()));
}

// A test case where the consecutive elements of the input shape belonging to
// the same layout part are not in descending order.
TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensionsWrongInputLayout) {
  // Same physical layout as in AlignLayoutsWithoutTrivialDimensions, except
  // that the first two dimension numbers are exchanged.
  Shape input = ShapeUtil::make_shape_with_layout({3, 8, 5, 7, 11},
                                                  {2, 3, 1, 0, 4});
  auto aligned_shape = ShapeUtil::align_layouts(
      input, ShapeUtil::make_shape({4, 3, 2, 7, 5, 11}));
  EXPECT_FALSE(aligned_shape);
}

// A test case where the physical layout of the input shape does not place all
// dimensions that belong to the same alignment part consecutively.
TEST(AlignmentTest,
     AlignLayoutsWithoutTrivialDimensionsNonConsecutiveAlignmentPart) {
  Shape input = ShapeUtil::make_shape_with_layout({3, 8, 5, 7, 11},
                                                  {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::align_layouts(
      input, ShapeUtil::make_shape({4, 3, 2, 5, 77}));
  EXPECT_FALSE(aligned_shape);
}

}  // namespace
}  // namespace hice
