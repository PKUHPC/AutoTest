#include "hice/core/tensor.h"
#include "hice/core/shape_util.h"
#include "hice/basic/factories.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ElementsAre;
using ::testing::ContainerEq;
using ::testing::IsEmpty;
using ::testing::Each;

TEST(TensorConstructorTest, DifferentScalarType) {
  // bool tensor
  Tensor bool_tensor = full({2, 2}, false, device(kCPU).dtype(kBool));
  for(int i = 0; i < bool_tensor.size(); ++i) {
    ASSERT_EQ(bool_tensor.data<bool>()[i], false);
  }
  Tensor bool_tensor2 =
      full({3, 3, 3}, true, device(kCUDA).dtype(kBool)).to(kCPU);
  for (int i = 0; i < bool_tensor2.size(); ++i) {
    ASSERT_EQ(bool_tensor2.data<bool>()[i], true);
  }

  // int32 tensor
  Tensor int32_tensor = full({2}, 1, device(kCPU).dtype(kInt32));
  for(int i = 0; i < int32_tensor.size(); ++i) {
    ASSERT_EQ(int32_tensor.data<int32_t>()[i], 1);
  }
  Tensor int32_tensor2 =
      full({3, 3, 3, 3}, 1, device(kCUDA).dtype(kInt32)).to(kCPU);
  for (int i = 0; i < int32_tensor2.size(); ++i) {
    ASSERT_EQ(int32_tensor2.data<int32_t>()[i], 1);
  }

  // float tensor
  Tensor float_tensor = full({2}, 2, device(kCPU).dtype(kFloat));
  for(int i = 0; i < float_tensor.size(); ++i) {
    ASSERT_EQ(float_tensor.data<float>()[i], 2);
  }
  Tensor float_tensor2 =
      full({3, 3, 3, 3}, 2, device(kCUDA).dtype(kFloat)).to(kCPU);
  for (int i = 0; i < float_tensor2.size(); ++i) {
    ASSERT_EQ(float_tensor2.data<float>()[i], 2);
  }
}

TEST(TensorConstructorTest, DifferentDimensions) {
  Tensor scalar = Tensor({}, dtype(kFloat).device(kCPU));
  EXPECT_TRUE(scalar.is_same(scalar));
  EXPECT_EQ(scalar.size(), 1);
  EXPECT_EQ(scalar.offset(), 0);
  EXPECT_EQ(scalar.data_type(), DataType::make<float>());
  EXPECT_EQ(scalar.device(), Device(kCPU));
  EXPECT_EQ(scalar.shape(), ShapeUtil::make_shape({}));
  EXPECT_THAT(scalar.strides(), IsEmpty());

  Tensor vector = Tensor({0}, dtype(kFloat).device(kCPU));
  EXPECT_TRUE(vector.is_same(vector));
  EXPECT_EQ(vector.size(), 0);
  EXPECT_EQ(vector.offset(), 0);
  EXPECT_EQ(vector.data_type(), DataType::make<float>());
  EXPECT_EQ(vector.device(), Device(kCPU));
  EXPECT_EQ(vector.shape(), ShapeUtil::make_shape({0}));
  EXPECT_THAT(vector.strides(), ElementsAre(1));

  Tensor vector2 = Tensor({1}, dtype(kFloat).device(kCPU));
  EXPECT_TRUE(vector2.is_same(vector2));
  EXPECT_EQ(vector2.size(), 1);
  EXPECT_EQ(vector2.offset(), 0);
  EXPECT_EQ(vector2.data_type(), DataType::make<float>());
  EXPECT_EQ(vector2.device(), Device(kCPU));
  EXPECT_EQ(vector2.shape(), ShapeUtil::make_shape({1}));
  EXPECT_THAT(vector2.strides(), ElementsAre(1));

  Tensor matrix = Tensor({2, 3}, dtype(kFloat).device(kCPU));
  EXPECT_TRUE(matrix.is_same(matrix));
  EXPECT_EQ(matrix.size(), 6);
  EXPECT_EQ(matrix.offset(), 0);
  EXPECT_EQ(matrix.data_type(), DataType::make<float>());
  EXPECT_EQ(matrix.device(), Device(kCPU));
  EXPECT_EQ(matrix.shape(), ShapeUtil::make_shape({2, 3}));
  EXPECT_THAT(matrix.strides(), ElementsAre(3, 1));

  Tensor tensor = Tensor({4, 5, 6}, dtype(kFloat).device(kCPU));
  EXPECT_TRUE(tensor.is_same(tensor));
  EXPECT_FALSE(tensor.is_same(matrix));
  EXPECT_FALSE(matrix.is_same(tensor));
  EXPECT_EQ(tensor.size(), 120);
  EXPECT_EQ(tensor.offset(), 0);
  EXPECT_EQ(tensor.data_type(), DataType::make<float>());
  EXPECT_EQ(tensor.device(), Device(kCPU));
  EXPECT_EQ(tensor.shape(), ShapeUtil::make_shape({4, 5, 6}));
  EXPECT_THAT(tensor.strides(), ElementsAre(30, 6, 1));

}

TEST(TensorToDeviceTest, CUDAToCPU) {
  Tensor d_tensor =
      hice::full({20, 30, 40}, 1, dtype(kDouble).device(kCUDA));
  Tensor h_tensor = d_tensor.to(kCPU); 
  hice::ArrayRef<double> data(h_tensor.mutable_data<double>(), h_tensor.size());
  EXPECT_THAT(data, Each(1));
}

TEST(TensorToDeviceTest, CPUToCUDA) {
  Tensor tensor =
      hice::full({20, 30, 40}, 2, dtype(kDouble).device(kCPU));
  Tensor d_tensor = tensor.to(kCUDA); 
  Tensor h_tensor = d_tensor.to(kCPU); 
  hice::ArrayRef<double> data(h_tensor.mutable_data<double>(), h_tensor.size());
  EXPECT_THAT(data, Each(2));
}

}  // namespace
}  // namespace hice
