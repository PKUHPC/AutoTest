#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::Each;

#if 0
TEST(TensorFactoriesTest, Rand) {
  Tensor tensor_cpu = rand_normal({10}, 0, 1, dtype(kFloat).device(kCPU));
  Tensor tensor_cuda = rand_normal({10}, 0, 1, dtype(kFloat).device(kCUDA));
  TensorPrinter tp;
  tp.print(tensor_cpu);
  tp.print(tensor_cuda);
}
#endif

TEST(TensorFactoriesTest, FullBoolScalarOnCPU) {
  Tensor scalar = full({}, true, dtype(kBool).device(kCPU));
  hice::ArrayRef<bool> scalar_data(scalar.mutable_data<bool>(), scalar.size());
  EXPECT_THAT(scalar_data, Each(true));

  Tensor vector = full({1}, false, dtype(kBool).device(kCPU));
  hice::ArrayRef<bool> vector_data(vector.mutable_data<bool>(), vector.size());
  EXPECT_THAT(vector_data, Each(false));

  Tensor matrix = full({2, 3}, true, dtype(kBool).device(kCPU));
  hice::ArrayRef<bool> matrix_data(matrix.mutable_data<bool>(), matrix.size());
  EXPECT_THAT(matrix_data, Each(true));

  Tensor tensor = full({4, 5, 6}, false, dtype(kBool).device(kCPU));
  hice::ArrayRef<bool> tensor_data(tensor.mutable_data<bool>(), tensor.size());
  EXPECT_THAT(tensor_data, Each(false));
}

TEST(TensorFactoriesTest, FullInt32ScalarOnCPU) {
  Tensor scalar = full({}, 1, dtype(kInt32).device(kCPU));
  hice::ArrayRef<int32_t> scalar_data(scalar.mutable_data<int32_t>(), scalar.size());
  EXPECT_THAT(scalar_data, Each(1));

  Tensor vector = full({1}, 2, dtype(kInt32).device(kCPU));
  hice::ArrayRef<int32_t> vector_data(vector.mutable_data<int32_t>(), vector.size());
  EXPECT_THAT(vector_data, Each(2));

  Tensor matrix = full({2, 3}, 3, dtype(kInt32).device(kCPU));
  hice::ArrayRef<int32_t> matrix_data(matrix.mutable_data<int32_t>(), matrix.size());
  EXPECT_THAT(matrix_data, Each(3));

  Tensor tensor = full({4, 5, 6}, 4, dtype(kInt32).device(kCPU));
  hice::ArrayRef<int32_t> tensor_data(tensor.mutable_data<int32_t>(), tensor.size());
  EXPECT_THAT(tensor_data, Each(4));
}

TEST(TensorFactoriesTest, FullFloatScalarOnCPU) {
  Tensor scalar = full({}, 1, dtype(kFloat).device(kCPU));
  hice::ArrayRef<float> scalar_data(scalar.mutable_data<float>(), scalar.size());
  EXPECT_THAT(scalar_data, Each(1));

  Tensor vector = full({1}, 2, dtype(kFloat).device(kCPU));
  hice::ArrayRef<float> vector_data(vector.mutable_data<float>(), vector.size());
  EXPECT_THAT(vector_data, Each(2));

  Tensor matrix = full({2, 3}, 3, dtype(kFloat).device(kCPU));
  hice::ArrayRef<float> matrix_data(matrix.mutable_data<float>(), matrix.size());
  EXPECT_THAT(matrix_data, Each(3));

  Tensor tensor = full({4, 5, 6}, 4, dtype(kFloat).device(kCPU));
  hice::ArrayRef<float> tensor_data(tensor.mutable_data<float>(), tensor.size());
  EXPECT_THAT(tensor_data, Each(4));
}

}  // namespace
}  // namespace hice
