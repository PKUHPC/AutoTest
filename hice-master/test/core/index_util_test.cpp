#include "hice/core/index_util.h"
#include "hice/core/shape_util.h"

#include "gtest/gtest.h"

using namespace hice;

void set_minor_to_major_layout(Shape &shape, std::vector<int64_t> dimensions) {
  shape.mutable_layout().clear_minor_to_major();
  for (auto dimension : dimensions) {
    shape.mutable_layout().add_minor_to_major(dimension);
  }
}

TEST(IndexUtilTest, VectorIndexing) {
  // Vectors are trivially laid out and the linear offset should always be the
  // same as the "multidimensional" index.
  Shape vector_shape = ShapeUtil::make_shape({100});
  EXPECT_EQ(42, IndexUtil::multi_index_to_offset(vector_shape, {42}));
  std::vector<int64_t> multi_index =
      IndexUtil::offset_to_multi_index(vector_shape, 42);
  EXPECT_EQ(1, multi_index.size());
  EXPECT_EQ(42, multi_index[0]);
}

TEST(IndexUtilTest, MatrixIndexingColumnMajor) {
  // Set layout to [0, 1]. That is, column major.
  Shape matrix_shape_01 = ShapeUtil::make_shape({10, 20});
  set_minor_to_major_layout(matrix_shape_01, {0, 1});

  // If index is {a, b} then linear index should be: a + b * 10
  EXPECT_EQ(0, IndexUtil::multi_index_to_offset(matrix_shape_01, {0, 0}));
  EXPECT_EQ(199, IndexUtil::multi_index_to_offset(matrix_shape_01, {9, 19}));
  EXPECT_EQ(53, IndexUtil::multi_index_to_offset(matrix_shape_01, {3, 5}));
  EXPECT_EQ(std::vector<int64_t>({3, 5}),
            IndexUtil::offset_to_multi_index(matrix_shape_01, 53));
}

TEST(IndexUtilTest, MatrixIndexingRowMajor) {
  // Set layout to [1, 0]. That is, row major.
  Shape matrix_shape_10 = ShapeUtil::make_shape({10, 20});
  set_minor_to_major_layout(matrix_shape_10, {1, 0});

  // If index is {a, b} then linear index should be: a * 20 + b
  EXPECT_EQ(0, IndexUtil::multi_index_to_offset(matrix_shape_10, {0, 0}));
  EXPECT_EQ(199, IndexUtil::multi_index_to_offset(matrix_shape_10, {9, 19}));
  EXPECT_EQ(65, IndexUtil::multi_index_to_offset(matrix_shape_10, {3, 5}));
  EXPECT_EQ(std::vector<int64_t>({3, 5}),
            IndexUtil::offset_to_multi_index(matrix_shape_10, 65));
}

TEST(IndexUtilTest, ThreeDArrayIndexing210) {
  // Set layout to [2, 1, 0]. That is, column major.
  Shape shape_210 = ShapeUtil::make_shape({10, 20, 30});
  set_minor_to_major_layout(shape_210, {2, 1, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b * 30 + c
  EXPECT_EQ(1957, IndexUtil::multi_index_to_offset(shape_210, {3, 5, 7}));
  EXPECT_EQ(5277, IndexUtil::multi_index_to_offset(shape_210, {8, 15, 27}));
}

TEST(IndexUtilTest, ThreeDArrayIndexing120) {
  // Set layout to [1, 2, 0]
  Shape shape_120 = ShapeUtil::make_shape({10, 20, 30});
  set_minor_to_major_layout(shape_120, {1, 2, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b + c * 20
  EXPECT_EQ(1945, IndexUtil::multi_index_to_offset(shape_120, {3, 5, 7}));
  EXPECT_EQ(5355, IndexUtil::multi_index_to_offset(shape_120, {8, 15, 27}));
}

TEST(IndexUtilTest, FourDArrayIndexing3210) {
  // Set layout to [3, 2, 1,0]. That is, column major.
  Shape shape_3210 = ShapeUtil::make_shape({10, 20, 30, 40});
  set_minor_to_major_layout(shape_3210, {3, 2, 1, 0});

  // If index is {a, b, c, d} then linear index should be:
  // a * 20 * 30 * 40 + b * 30 * 40 + c * 40 + d
  EXPECT_EQ(78289, IndexUtil::multi_index_to_offset(shape_3210, {3, 5, 7, 9}));
  EXPECT_EQ(211113,
            IndexUtil::multi_index_to_offset(shape_3210, {8, 15, 27, 33}));
}

TEST(IndexUtilTest, LinearToMultiToLinear) {
  // Verify that converting a linear index to a multidimensional index and back
  // always returns the same value for different crazy shapes.  Shape has
  // 1440000000 elements. Inputs are randomly-ish selected.
  std::vector<int64_t> offsets = {0,        1439999999, 1145567336,
                                43883404, 617295214,  1117613654};

  std::vector<std::vector<int64_t>> minor_to_major_orders;
  minor_to_major_orders.push_back({6, 5, 4, 3, 2, 1, 0});
  minor_to_major_orders.push_back({0, 1, 2, 3, 4, 5, 6});
  minor_to_major_orders.push_back({4, 5, 1, 2, 6, 0, 3});

  for (auto minor_to_major_order : minor_to_major_orders) {
    Shape shape = ShapeUtil::make_shape({10, 20, 30, 40, 30, 20, 10});
    set_minor_to_major_layout(shape, minor_to_major_order);
    for (auto offset : offsets) {
      std::vector<int64_t> multi_index =
          IndexUtil::offset_to_multi_index(shape, offset);
      EXPECT_EQ(offset, IndexUtil::multi_index_to_offset(shape, multi_index));
    }
  }
}

TEST(IndexUtilTest, Last_Next_MultiIndex) {

  std::vector<std::vector<int64_t>> minor_to_major_orders;
  minor_to_major_orders.push_back({5, 4, 3, 2, 1, 0});
  minor_to_major_orders.push_back({0, 1, 2, 3, 4, 5});
  minor_to_major_orders.push_back({4, 5, 1, 2, 0, 3});

  int64_t offset;
  for (auto minor_to_major_order : minor_to_major_orders) {
    std::vector<int64_t> dims = {2, 3, 4, 7, 6, 1};
    Shape shape = ShapeUtil::make_shape(dims);
    set_minor_to_major_layout(shape, minor_to_major_order);

    offset = 0;
    int64_t size = ShapeUtil::get_num_items(shape);
    std::vector<int64_t> multi_index(shape.rank(), 0);
    while (offset < size - 1) {
      ++offset;
      std::vector<int64_t> multi_index_from_offset =
          IndexUtil::offset_to_multi_index(shape, offset);
      IndexUtil::next_multi_index(shape, multi_index);
      for (int i = 0 ; i < shape.rank(); ++i) {
        EXPECT_EQ(multi_index_from_offset[i], multi_index[i]);
      }
    }

    offset = size - 1;
    for (int j = 0; j < shape.rank(); ++j) {
      multi_index[j] = dims[j] - 1;
    }
    while (offset > 0 ) {
      --offset;
      std::vector<int64_t> multi_index_from_offset =
          IndexUtil::offset_to_multi_index(shape, offset);
      IndexUtil::last_multi_index(shape, multi_index);
      for (int i = 0 ; i < shape.rank(); ++i) {
        EXPECT_EQ(multi_index_from_offset[i], multi_index[i]);
      }
    }
  }
}

#if 0
TEST(IndexUtilTest, BumpIndices2x2) {
  auto shape = ShapeUtil::make_shape({2, 2});
  std::vector<int64_t> indices = {0, 0};
  EXPECT_TRUE(IndexUtil::bump_indices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(0, 1));
  EXPECT_TRUE(IndexUtil::bump_indices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 0));
  EXPECT_TRUE(IndexUtil::bump_indices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 1));
  EXPECT_FALSE(IndexUtil::bump_indices(shape, absl::MakeSpan(indices)));
}
#endif