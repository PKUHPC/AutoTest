#include "hice/core/tensor.h"
#include "hice/core/shape_util.h"
#include "hice/core/index_util.h"
#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/resize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::Each;

TEST(ResizeTest, ShrinkedDimensions) {
  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  hice::resize_(vector, {});
  EXPECT_EQ(vector.ndim(), 0);
  EXPECT_EQ(vector.size(), 1);
  hice::ArrayRef<float> vector_data(vector.mutable_data<float>(), vector.size());
  EXPECT_THAT(vector_data, Each(1));

  Tensor matrix = hice::full({2, 3}, 1, dtype(kFloat).device(kCPU));
  hice::resize_(matrix, {2});
  EXPECT_EQ(matrix.ndim(), 1);
  EXPECT_EQ(matrix.size(), 2);
  hice::ArrayRef<float> matrix_data(matrix.mutable_data<float>(), matrix.size());
  EXPECT_THAT(matrix_data, Each(1));

  Tensor tensor = hice::full({2, 3, 4}, 1, dtype(kFloat).device(kCPU));
  hice::resize_(tensor, {2, 3});
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.size(), 6);
  hice::ArrayRef<float> tensor_data(tensor.mutable_data<float>(), tensor.size());
  EXPECT_THAT(tensor_data, Each(1));
}

TEST(ResizeTest, SameDimensions) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  int64_t old_scalar_size = scalar.size();
  hice::resize_(scalar, {});
  EXPECT_EQ(scalar.ndim(), 0);
  EXPECT_EQ(scalar.size(), 1);
  int64_t new_scalar_size = scalar.size();
  hice::ArrayRef<float> old_scalar_data(scalar.mutable_data<float>(),
                                    old_scalar_size);
  hice::ArrayRef<float> new_scalar_data(
      scalar.mutable_data<float>() + old_scalar_size,
      new_scalar_size - old_scalar_size);
  EXPECT_THAT(old_scalar_data, Each(1));
  EXPECT_THAT(new_scalar_data, Each(0));

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  int64_t old_vector_size = vector.size();
  hice::resize_(vector, {3});
  EXPECT_EQ(vector.ndim(), 1);
  EXPECT_EQ(vector.size(), 3);
  int64_t new_vector_size = vector.size();
  hice::ArrayRef<float> old_vector_data(vector.mutable_data<float>(),
                                    old_vector_size);
  hice::ArrayRef<float> new_vector_data(
      vector.mutable_data<float>() + old_vector_size,
      new_vector_size - old_vector_size);
  EXPECT_THAT(old_vector_data, Each(1));
  EXPECT_THAT(new_vector_data, Each(0));
  
  Tensor matrix = hice::full({2, 3}, 1, dtype(kFloat).device(kCPU));
  int64_t old_matrix_size = matrix.size();
  hice::resize_(matrix, {4, 5});
  EXPECT_EQ(matrix.ndim(), 2);
  EXPECT_EQ(matrix.size(), 20);
  int64_t new_matrix_size = matrix.size();
  hice::ArrayRef<float> old_matrix_data(matrix.mutable_data<float>(),
                                    old_matrix_size);
  hice::ArrayRef<float> new_matrix_data(
      matrix.mutable_data<float>() + old_matrix_size,
      new_matrix_size - old_matrix_size);
  EXPECT_THAT(old_matrix_data, Each(1));
  EXPECT_THAT(new_matrix_data, Each(0));

  Tensor tensor = hice::full({2, 3, 4}, 1, dtype(kFloat).device(kCPU));
  int64_t old_tensor_size = tensor.size();
  hice::resize_(tensor, {4, 5, 6});
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.size(), 120);
  int64_t new_tensor_size = tensor.size();
  hice::ArrayRef<float> old_tensor_data(tensor.mutable_data<float>(),
                                    old_tensor_size);
  hice::ArrayRef<float> new_tensor_data(
      tensor.mutable_data<float>() + old_tensor_size,
      new_tensor_size - old_tensor_size);
  EXPECT_THAT(old_tensor_data, Each(1));
  EXPECT_THAT(new_tensor_data, Each(0));
}

TEST(ResizeTest, ExpandedDimensions) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  int64_t old_scalar_size = scalar.size();
  hice::resize_(scalar, {2});
  EXPECT_EQ(scalar.ndim(), 1);
  EXPECT_EQ(scalar.size(), 2);
  int64_t new_scalar_size = scalar.size();
  hice::ArrayRef<float> old_scalar_data(scalar.mutable_data<float>(),
                                    old_scalar_size);
  hice::ArrayRef<float> new_scalar_data(
      scalar.mutable_data<float>() + old_scalar_size,
      new_scalar_size - old_scalar_size);
  EXPECT_THAT(old_scalar_data, Each(1));
  EXPECT_THAT(new_scalar_data, Each(0));

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  int64_t old_vector_size = vector.size();
  hice::resize_(vector, {3, 4});
  EXPECT_EQ(vector.ndim(), 2);
  EXPECT_EQ(vector.size(), 12);
  int64_t new_vector_size = vector.size();
  hice::ArrayRef<float> old_vector_data(vector.mutable_data<float>(),
                                    old_vector_size);
  hice::ArrayRef<float> new_vector_data(
      vector.mutable_data<float>() + old_vector_size,
      new_vector_size - old_vector_size);
  EXPECT_THAT(old_vector_data, Each(1));
  EXPECT_THAT(new_vector_data, Each(0));
  
  Tensor matrix = hice::full({2, 3}, 1, dtype(kFloat).device(kCPU));
  int64_t old_matrix_size = matrix.size();
  hice::resize_(matrix, {3, 4, 5});
  EXPECT_EQ(matrix.ndim(), 3);
  EXPECT_EQ(matrix.size(), 60);
  int64_t new_matrix_size = matrix.size();
  hice::ArrayRef<float> old_matrix_data(matrix.mutable_data<float>(),
                                    old_matrix_size);
  hice::ArrayRef<float> new_matrix_data(
      matrix.mutable_data<float>() + old_matrix_size,
      new_matrix_size - old_matrix_size);
  EXPECT_THAT(old_matrix_data, Each(1));
  EXPECT_THAT(new_matrix_data, Each(0));

  Tensor tensor = hice::full({2, 3, 4}, 1, dtype(kFloat).device(kCPU));
  int64_t old_tensor_size = tensor.size();
  hice::resize_(tensor, {3, 4, 5, 6});
  EXPECT_EQ(tensor.ndim(), 4);
  EXPECT_EQ(tensor.size(), 360);
  int64_t new_tensor_size = tensor.size();
  hice::ArrayRef<float> old_tensor_data(tensor.mutable_data<float>(),
                                    old_tensor_size);
  hice::ArrayRef<float> new_tensor_data(
      tensor.mutable_data<float>() + old_tensor_size,
      new_tensor_size - old_tensor_size);
  EXPECT_THAT(old_tensor_data, Each(1));
  EXPECT_THAT(new_tensor_data, Each(0));
}

}
} // namespace hice