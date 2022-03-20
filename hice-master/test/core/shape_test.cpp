#include "hice/core/layout_util.h"
#include "hice/core/shape_util.h"
#include "hice/core/shape.h"

#include "gtest/gtest.h"

namespace hice {
namespace {

class ShapeTest : public ::testing::Test {
 protected:
  const Shape scalar_ = ShapeUtil::make_shape({});
  const Shape vector_ = ShapeUtil::make_shape({1});
  const Shape matrix_ = ShapeUtil::make_shape({1, 2});
  const Shape tensor_ = ShapeUtil::make_shape_with_layout({1, 2, 3}, {0, 1, 2});
};

TEST_F(ShapeTest, ShapeConstruction) {
  // Scalar shape test for scalar_ 
  EXPECT_EQ(scalar_.dimensions().size(), 0);
  EXPECT_EQ(scalar_.layout().type(), kDense);
  EXPECT_EQ(scalar_.layout().minor_to_major().size(),
            scalar_.dimensions().size());
  // Matrix shape test for vector_ 
  EXPECT_EQ(vector_.dimensions().size(), 1);
  for (int i = 0; i < vector_.rank(); ++i) {
    EXPECT_EQ(vector_.dimensions()[i], i + 1);
  }
  EXPECT_EQ(vector_.layout().type(), kDense);
  EXPECT_EQ(vector_.layout().minor_to_major().size(),
            vector_.dimensions().size());
  for (int i = 0; i < vector_.rank(); ++i) {
    EXPECT_EQ(vector_.layout().minor_to_major()[i], vector_.rank() - 1 - i);
  }
  // Matrix shape test for matrix_ 
  EXPECT_EQ(matrix_.dimensions().size(), 2);
  for (int i = 0; i < matrix_.rank(); ++i) {
    EXPECT_EQ(matrix_.dimensions()[i], i + 1);
  }
  EXPECT_EQ(matrix_.layout().type(), kDense);
  EXPECT_EQ(matrix_.layout().minor_to_major().size(),
            matrix_.dimensions().size());
  for (int i = 0; i < matrix_.rank(); ++i) {
    EXPECT_EQ(matrix_.layout().minor_to_major()[i], matrix_.rank() - 1 - i);
  }
  // Matrix shape test for tensor_ 
  EXPECT_EQ(tensor_.dimensions().size(), 3);
  for (int i = 0; i < tensor_.rank(); ++i) {
    EXPECT_EQ(tensor_.dimensions()[i], i + 1);
  }
  EXPECT_EQ(tensor_.layout().type(), kDense);
  EXPECT_EQ(tensor_.layout().minor_to_major().size(),
            tensor_.dimensions().size());
  for (int i = 0; i < tensor_.rank(); ++i) {
    EXPECT_EQ(tensor_.layout().minor_to_major()[i], i);
  }
}

TEST_F(ShapeTest, ShapeToString) {
  // Print without layout
  EXPECT_EQ("[]", scalar_.to_string());
  EXPECT_EQ("[1]", vector_.to_string());
  EXPECT_EQ("[1,2]", matrix_.to_string());
  EXPECT_EQ("[1,2,3]", tensor_.to_string());
  // Print with layout
  EXPECT_EQ("[]", scalar_.to_string(/*print_layout=*/true));
  EXPECT_EQ("[1]{0}", vector_.to_string(/*print_layout=*/true));
  EXPECT_EQ("[1,2]{1,0}", matrix_.to_string(/*print_layout=*/true));
  EXPECT_EQ("[1,2,3]{0,1,2}", tensor_.to_string(/*print_layout=*/true));
}

}  // namespace
}  // namespace hice
