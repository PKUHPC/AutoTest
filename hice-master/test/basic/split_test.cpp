#include "hice/basic/split.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::Each;

void TENSOR_EXPECT_EQ(Tensor tensor1, Tensor tensor2) {
  EXPECT_EQ(tensor1.size(), tensor2.size());
  // EXPECT_EQ(tensor1.offset(), tensor2.offset());
  EXPECT_EQ(tensor1.data_type(), tensor2.data_type());
  EXPECT_EQ(tensor1.device(), tensor2.device());
  EXPECT_EQ(tensor1.shape(), tensor2.shape());
  EXPECT_EQ(tensor1.strides(), tensor2.strides());
  Tensor tensor1_new =
      tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new =
      tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(tensor1_new.data<float>()[i], tensor2_new.data<float>()[i]);
  }
}

TEST(SplitTest, split) {
  Tensor tensor = hice::full({6, 6, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor sub_res = hice::full({3, 6, 4}, 1, dtype(kFloat).device(kCPU));
  std::vector<Tensor> sub_tensors = hice::split(tensor, 0, 2);
  for (int i = 0; i < sub_tensors.size(); ++i) {
    TENSOR_EXPECT_EQ(sub_tensors[i], sub_res);
  }
}

TEST(SplitTest, split_with_sizes) {
  Tensor tensor = hice::full({10, 15, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor sub_res = hice::full({10, 3, 4}, 1, dtype(kFloat).device(kCPU));
  std::vector<int64_t> sizes = {3, 3, 3, 3, 3};
  std::vector<Tensor> sub_tensors = hice::split_with_sizes(tensor, 1, sizes);
  for (int i = 0; i < sub_tensors.size(); ++i) {
    TENSOR_EXPECT_EQ(sub_tensors[i], sub_res);
  }
}

}  // namespace
}  // namespace hice