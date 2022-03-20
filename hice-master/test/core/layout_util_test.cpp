#include "hice/core/shape_util.h"
#include "hice/core/layout_util.h"

#include "gtest/gtest.h"

namespace hice {
namespace {

class LayoutUtilTest : public ::testing::Test {
 protected:
  Shape make_shape_with_layout(ConstIntArrayRef dimensions,
                               ConstIntArrayRef minor_to_major) {
    Shape shape = ShapeUtil::make_shape(dimensions);
    shape.mutable_layout() = LayoutUtil::make_layout(minor_to_major);
    return shape;
  }
};

TEST_F(LayoutUtilTest, CopyLayoutArray) {
  Shape src = make_shape_with_layout({2, 3}, {0, 1});
  Shape dst = make_shape_with_layout({2, 3}, {1, 0});

  EXPECT_FALSE(LayoutUtil::layouts_in_shapes_equal(src, dst));
  EXPECT_TRUE(LayoutUtil::copy_layout_between_shapes(src, dst));
  EXPECT_TRUE(LayoutUtil::layouts_in_shapes_equal(src, dst));

  // Should work if destination has no layout.
  dst.clear_layout();
  EXPECT_FALSE(LayoutUtil::layouts_in_shapes_equal(src, dst));
  EXPECT_TRUE(LayoutUtil::copy_layout_between_shapes(src, dst));
  EXPECT_TRUE(LayoutUtil::layouts_in_shapes_equal(src, dst));

  // If source is cleared, then destination should be cleared.
  src.clear_layout();
  EXPECT_FALSE(LayoutUtil::layouts_in_shapes_equal(src, dst));
  EXPECT_TRUE(dst.has_layout());
  EXPECT_TRUE(LayoutUtil::copy_layout_between_shapes(src, dst));
  EXPECT_TRUE(LayoutUtil::layouts_in_shapes_equal(src, dst));
  EXPECT_FALSE(dst.has_layout());
}

TEST_F(LayoutUtilTest, IsDefaultLayout) {
  Shape scalar_shape = make_shape_with_layout({}, {});
  EXPECT_TRUE(LayoutUtil::is_default_layout(scalar_shape));

  Shape vector_shape = make_shape_with_layout({2}, {0});
  EXPECT_TRUE(LayoutUtil::is_default_layout(vector_shape));

  Shape matrix_shape = make_shape_with_layout({2, 3}, {0, 1});
  Shape matrix_shape2 = make_shape_with_layout({2, 3}, {1, 0});
  EXPECT_FALSE(LayoutUtil::is_default_layout(matrix_shape));
  EXPECT_TRUE(LayoutUtil::is_default_layout(matrix_shape2));

  Shape tensor_shape = make_shape_with_layout({2, 3, 4}, {2, 0, 1});
  Shape tensor_shape2 = make_shape_with_layout({2, 3, 4}, {2, 1, 0});
  EXPECT_FALSE(LayoutUtil::is_default_layout(tensor_shape));
  EXPECT_TRUE(LayoutUtil::is_default_layout(tensor_shape2));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleSameRank) {
  Shape src = make_shape_with_layout({123, 42, 7}, {2, 0, 1});
  Shape dst = make_shape_with_layout({2, 3, 5}, {1, 0});
  ASSERT_TRUE(LayoutUtil::copy_layout_between_shapes(src, dst));
  EXPECT_TRUE(LayoutUtil::layouts_in_shapes_equal(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleDifferentRank) {
  Shape src = make_shape_with_layout({123, 42, 7}, {2, 0, 1});
  Shape dst = make_shape_with_layout({2, 3}, {1, 0});
  EXPECT_FALSE(LayoutUtil::copy_layout_between_shapes(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutBogusLayout) {
  Shape src = ShapeUtil::make_shape({2, 3});
  Shape dst = ShapeUtil::make_shape({2, 3});
  // Set layout to invalid value.
  src.mutable_layout() = LayoutUtil::make_layout({1, 2, 3, 4});
  EXPECT_FALSE(LayoutUtil::copy_layout_between_shapes(src, dst));
}

TEST_F(LayoutUtilTest, DefaultLayoutGettersMajorToMinor) {
  EXPECT_TRUE(LayoutUtil::equal(LayoutUtil::make_layout({1, 0}),
                                LayoutUtil::get_default_layout_for_rank(2)));
  EXPECT_TRUE(LayoutUtil::equal(LayoutUtil::make_layout({2, 1, 0}),
                                LayoutUtil::get_default_layout_for_rank(3)));
  EXPECT_TRUE(LayoutUtil::equal(LayoutUtil::make_layout({3, 2, 1, 0}),
                                LayoutUtil::get_default_layout_for_rank(4)));
  EXPECT_TRUE(
      LayoutUtil::equal(LayoutUtil::make_layout({4, 3, 2, 1, 0}),
                        LayoutUtil::get_default_layout_for_shape(
                            ShapeUtil::make_shape({10, 20, 30, 15, 25}))));
}

TEST_F(LayoutUtilTest, ValidateLayout_ValidArrayLayout) {
  Shape shape = ShapeUtil::make_shape_with_layout({2, 3}, {0, 1});
  auto status = LayoutUtil::validate_layout_in_shape(
      shape, /*allow_missing_layouts=*/false);
  EXPECT_TRUE(status);
  status = LayoutUtil::validate_layout_in_shape(shape,
                                                /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status);
}

TEST_F(LayoutUtilTest, ValidateLayout_InvalidArrayLayout) {
  Shape shape = ShapeUtil::make_shape({2, 3});
  shape.mutable_layout() = LayoutUtil::make_layout({0, 1, 2});
  auto status = LayoutUtil::validate_layout_in_shape(
      shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status);
  status = LayoutUtil::validate_layout_in_shape(shape,
                                                /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status);
}

TEST_F(LayoutUtilTest, ValidateLayout_MissingArrayLayout) {
  Shape shape = ShapeUtil::make_shape({2, 3});
  LayoutUtil::clear_layout(shape);
  auto status = LayoutUtil::validate_layout_in_shape(
      shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status);
  status = LayoutUtil::validate_layout_in_shape(shape,
                                                /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status);
}

}  // namespace
}  // namespace hice
