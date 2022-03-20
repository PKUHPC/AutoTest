#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/basic/one_hot.h"
#include "hice/nn/cross_entropy.h"
#include "hice/nn/sparse_cross_entropy.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ContainerEq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(SparseCrossEntropyForwardTest, CompareCUDAWithCPU) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({8, 10}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor target =
      rand_uniform({prob.dim(0)}, 0, prob.dim(1), dtype(kInt64).device(kCPU));

  Tensor cpu_loss = sparse_cross_entropy_fwd(prob, target, hice::nullopt, 1);
  hice::ArrayRef<float> cpu_loss_data(cpu_loss.mutable_data<float>(), cpu_loss.size());

  Tensor cuda_loss = sparse_cross_entropy_fwd(prob.to(kCUDA), target.to(kCUDA),
                                              hice::nullopt, 1).to(kCPU);
  hice::ArrayRef<float> cuda_loss_data(cuda_loss.mutable_data<float>(), cuda_loss.size());

  EXPECT_THAT(cuda_loss_data, Pointwise(FloatEq(), cpu_loss_data));
  for (int i = 0; i < cuda_loss_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cuda_loss_data[i] - cpu_loss_data[i]) < 0.1);
  }

}

TEST(SparseCrossEntropyBackwardTest, CompareCUDAWithCPU) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({10, 12}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor grad_loss =
      rand_uniform({prob.dim(1)}, 0, 4, dtype(kFloat).device(kCPU));
  Tensor target =
      rand_uniform({prob.dim(1)}, 0, prob.dim(0), dtype(kInt64).device(kCPU));

  Tensor cpu_grad_prob =
      sparse_cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, 0);
  hice::ArrayRef<float> cpu_grad_prob_data(cpu_grad_prob.mutable_data<float>(),
                                       cpu_grad_prob.size());

  Tensor cuda_grad_prob =
      sparse_cross_entropy_bwd(prob.to(kCUDA), target.to(kCUDA), hice::nullopt,
                               grad_loss.to(kCUDA), 0).to(kCPU);
  hice::ArrayRef<float> cuda_grad_prob_data(cuda_grad_prob.mutable_data<float>(),
                                        cuda_grad_prob.size());

  EXPECT_THAT(cuda_grad_prob_data, Pointwise(FloatEq(), cpu_grad_prob_data));
  for (int i = 0; i < cuda_grad_prob_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cuda_grad_prob_data[i] - cpu_grad_prob_data[i]) < 0.1);
  }

  //tp.print(prob);
  //tp.print(grad_loss);
  //tp.print(sparse_target);
  //tp.print(target);
  //tp.print(sparse_grad_prob);
  //tp.print(grad_prob);
}

TEST(SparseCrossEntropyForwardTest, CompareSparseXenWithXen) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({5, 6}, 0, 1, dtype(kFloat).device(kCUDA));
  Tensor sparse_target =
      rand_uniform({prob.dim(0)}, 0, prob.dim(1), dtype(kInt64).device(kCPU))
          .to(kCUDA);
  Tensor sparse_loss =
      sparse_cross_entropy_fwd(prob, sparse_target, hice::nullopt, -1).to(kCPU);
  hice::ArrayRef<float> sparse_data(sparse_loss.mutable_data<float>(), sparse_loss.size());
  Tensor target = one_hot(sparse_target, prob.dim(1), -1).to(kFloat);
  Tensor loss = cross_entropy_fwd(prob, target, hice::nullopt, -1).to(kCPU);
  hice::ArrayRef<float> data(loss.mutable_data<float>(), loss.size());
  EXPECT_THAT(sparse_data, Pointwise(FloatEq(), data));
  for (int i = 0; i < sparse_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data[i] - data[i]) < 0.1);
  }

  Tensor prob2 = rand_uniform({7, 2, 4}, 0, 10, dtype(kFloat).device(kCUDA));
  Tensor sparse_target2 =
      rand_uniform({prob2.dim(1), prob2.dim(2)}, 0, prob2.dim(0), dtype(kInt64).device(kCPU)).to(kCUDA);
  Tensor sparse_loss2 =
      sparse_cross_entropy_fwd(prob2, sparse_target2, hice::nullopt, 0)
          .to(kCPU);
  hice::ArrayRef<float> sparse_data2(sparse_loss2.mutable_data<float>(),
                                 sparse_loss2.size());
  Tensor target2 = one_hot(sparse_target2, prob2.dim(0), 0).to(kFloat);
  Tensor loss2 = cross_entropy_fwd(prob2, target2, hice::nullopt, 0).to(kCPU);
  hice::ArrayRef<float> data2(loss2.mutable_data<float>(), loss2.size());
  EXPECT_THAT(sparse_data2, Pointwise(FloatEq(), data2));
  for (int i = 0; i < sparse_data2.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data2[i] - data2[i]) < 0.1);
  }

  Tensor prob3 = rand_uniform({8, 3, 4}, 0, 10, dtype(kFloat).device(kCUDA));
  Tensor sparse_target3 =
      rand_uniform({prob3.dim(0), prob3.dim(2)}, 0, prob3.dim(1), dtype(kInt64).device(kCPU)).to(kCUDA);
  Tensor sparse_loss3 =
      sparse_cross_entropy_fwd(prob3, sparse_target3, hice::nullopt, 1)
          .to(kCPU);
  hice::ArrayRef<float> sparse_data3(sparse_loss3.mutable_data<float>(),
                                 sparse_loss3.size());
  Tensor target3 = one_hot(sparse_target3, prob3.dim(1), 1).to(kFloat);
  Tensor loss3 = cross_entropy_fwd(prob3, target3, hice::nullopt, 1).to(kCPU);
  hice::ArrayRef<float> data3(loss3.mutable_data<float>(), loss3.size());
  EXPECT_THAT(sparse_data3, Pointwise(FloatEq(), data3));
  for (int i = 0; i < sparse_data3.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data3[i] - data3[i]) < 0.1);
  }
}

TEST(SparseCrossEntropyBackwardTest, CompareSparseXenWithXen) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({10, 6}, 0, 10, dtype(kFloat).device(kCUDA));
  Tensor grad_loss =
      rand_uniform({prob.dim(0)}, 0, 1, dtype(kFloat).device(kCUDA));
  Tensor sparse_target =
      rand_uniform({prob.dim(0)}, 0, prob.dim(1), dtype(kInt64).device(kCPU))
          .to(kCUDA);
  Tensor sparse_grad_prob =
      sparse_cross_entropy_bwd(prob, sparse_target, hice::nullopt, grad_loss,
                               -1).to(kCPU);
  hice::ArrayRef<float> sparse_data(sparse_grad_prob.mutable_data<float>(),
                                sparse_grad_prob.size());
  Tensor target = one_hot(sparse_target, prob.dim(1), -1).to(kFloat);
  Tensor grad_prob =
      cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, -1).to(kCPU);
  hice::ArrayRef<float> data(grad_prob.mutable_data<float>(), grad_prob.size());
  EXPECT_THAT(sparse_data, Pointwise(FloatEq(), data));
  for (int i = 0; i < sparse_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data[i] - data[i]) < 0.1);
  }

  Tensor prob2 = rand_uniform({4, 5}, 0, 10, dtype(kFloat).device(kCUDA));
  Tensor grad_loss2 =
      rand_uniform({prob2.dim(1)}, 0, 2, dtype(kFloat).device(kCUDA));
  Tensor sparse_target2 =
      rand_uniform({prob2.dim(1)}, 0, prob2.dim(0), dtype(kInt64).device(kCPU))
          .to(kCUDA);
  Tensor sparse_grad_prob2 =
      sparse_cross_entropy_bwd(prob2, sparse_target2, hice::nullopt, grad_loss2,
                               0).to(kCPU);
  hice::ArrayRef<float> sparse_data2(sparse_grad_prob2.mutable_data<float>(),
                                sparse_grad_prob2.size());
  Tensor target2 = one_hot(sparse_target2, prob2.dim(0), 0).to(kFloat);
  Tensor grad_prob2 =
      cross_entropy_bwd(prob2, target2, hice::nullopt, grad_loss2, 0).to(kCPU);
  hice::ArrayRef<float> data2(grad_prob2.mutable_data<float>(), grad_prob2.size());
  EXPECT_THAT(sparse_data2, Pointwise(FloatEq(), data2));
  for (int i = 0; i < sparse_data2.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data2[i] - data2[i]) < 0.1);
  }

  Tensor prob3 = rand_uniform({4, 9, 5}, 0, 10, dtype(kFloat).device(kCUDA));
  Tensor grad_loss3 =
      rand_uniform({prob3.dim(0), prob3.dim(2)}, 0, 3, dtype(kFloat).device(kCUDA));
  Tensor sparse_target3 =
      rand_uniform({prob3.dim(0), prob3.dim(2)}, 0, prob3.dim(1), dtype(kInt64).device(kCPU)).to(kCUDA);
  Tensor sparse_grad_prob3 =
      sparse_cross_entropy_bwd(prob3, sparse_target3, hice::nullopt, grad_loss3,
                               1).to(kCPU);
  hice::ArrayRef<float> sparse_data3(sparse_grad_prob3.mutable_data<float>(),
                                sparse_grad_prob3.size());
  Tensor target3 = one_hot(sparse_target3, prob3.dim(1), 1).to(kFloat);
  Tensor grad_prob3 =
      cross_entropy_bwd(prob3, target3, hice::nullopt, grad_loss3, 1).to(kCPU);
  hice::ArrayRef<float> data3(grad_prob3.mutable_data<float>(), grad_prob3.size());
  EXPECT_THAT(sparse_data3, Pointwise(FloatEq(), data3));
  for (int i = 0; i < sparse_data3.size(); ++ i) {
    EXPECT_TRUE(std::abs(sparse_data3[i] - data3[i]) < 0.1);
  }

  // tp.print(prob);
  // tp.print(grad_loss);
  // tp.print(sparse_target);
  // tp.print(sparse_grad_prob);
  // tp.print(grad_prob);
}

}  // namespace
}  // namespace hice