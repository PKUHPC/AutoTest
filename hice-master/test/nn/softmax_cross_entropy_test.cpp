#include "hice/basic/copy.h"
#include "hice/basic/factories.h"
#include "hice/basic/one_hot.h"
#include "hice/nn/softmax_cross_entropy.h"
#include "hice/nn/cross_entropy.h"
#include "hice/nn/softmax.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ContainerEq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(SoftmaxCrossEntropyForwardTest, CompareCUDAWithCPU) {
  Tensor logit = rand_uniform({8, 10}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor sparse_target =
      rand_uniform({logit.dim(0)}, 0, logit.dim(1), dtype(kInt64).device(kCPU));
  Tensor target = one_hot(sparse_target, logit.dim(1), 1).to(kFloat);

  Tensor cpu_loss = std::get<1>(
      softmax_cross_entropy_fwd(logit, target, hice::nullopt, 1));
  hice::ArrayRef<float> cpu_loss_data(cpu_loss.mutable_data<float>(), cpu_loss.size());

  Tensor cuda_loss =
      std::get<1>(softmax_cross_entropy_fwd(
                      logit.to(kCUDA), target.to(kCUDA), hice::nullopt, 1))
          .to(kCPU);
  hice::ArrayRef<float> cuda_loss_data(cuda_loss.mutable_data<float>(), cuda_loss.size());

//   EXPECT_THAT(cuda_loss_data, Pointwise(FloatEq(), cpu_loss_data));
  for (int i = 0; i < cuda_loss_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cuda_loss_data[i] - cpu_loss_data[i]) < 0.1);
  }

}

TEST(SoftmaxCrossEntropyBackwardTest, CompareCUDAWithCPU) {
  Tensor prob = rand_uniform({10, 12}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor sparse_target =
      rand_uniform({prob.dim(1)}, 0, prob.dim(0), dtype(kInt64).device(kCPU));
  Tensor target = one_hot(sparse_target, prob.dim(0), 0).to(kFloat);
  Tensor grad_loss =
      rand_uniform({prob.dim(1)}, 0, 4, dtype(kFloat).device(kCPU));

  Tensor cpu_grad_logit =
      softmax_cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, 0);
  hice::ArrayRef<float> cpu_grad_logit_data(cpu_grad_logit.mutable_data<float>(),
                                        cpu_grad_logit.size());

  Tensor cuda_grad_logit =
      softmax_cross_entropy_bwd(prob.to(kCUDA), target.to(kCUDA), hice::nullopt,
                                grad_loss.to(kCUDA), 0)
          .to(kCPU);
  hice::ArrayRef<float> cuda_grad_logit_data(cuda_grad_logit.mutable_data<float>(),
                                         cuda_grad_logit.size());

//   EXPECT_THAT(cuda_grad_logit_data, Pointwise(FloatEq(), cpu_grad_logit_data));
  for (int i = 0; i < cuda_grad_logit_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cuda_grad_logit_data[i] - cpu_grad_logit_data[i]) < 0.1);
  }

  //tp.print(prob);
  //tp.print(grad_loss);
  //tp.print(sparse_target);
  //tp.print(target);
  //tp.print(sparse_grad_prob);
  //tp.print(grad_prob);
}

TEST(SoftmaxCrossEntropyBackwardTest, CompareWithUnfusedKernels) {
  //TensorPrinter tp;
  Tensor prob = rand_uniform({2, 3}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor sparse_target =
      rand_uniform({prob.dim(1)}, 0, prob.dim(0), dtype(kInt64).device(kCPU));
  Tensor target = one_hot(sparse_target, prob.dim(0), 0).to(kFloat);
  Tensor grad_loss =
      rand_uniform({prob.dim(1)}, 0, 4, dtype(kFloat).device(kCPU));

  // cpu test
  Tensor cpu_grad_prob =
      cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, 0);
  Tensor cpu_grad_logit = softmax_bwd(prob, cpu_grad_prob, 0);
  hice::ArrayRef<float> cpu_grad_logit_data(cpu_grad_logit.mutable_data<float>(),
                                        cpu_grad_logit.size());

  Tensor cpu_grad_logit2 =
      softmax_cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, 0);
  hice::ArrayRef<float> cpu_grad_logit_data2(cpu_grad_logit2.mutable_data<float>(),
                                         cpu_grad_logit2.size());
//   EXPECT_THAT(cpu_grad_logit_data, Pointwise(FloatEq(), cpu_grad_logit_data2));
  for (int i = 0; i < cpu_grad_logit_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cpu_grad_logit_data[i] - cpu_grad_logit_data2[i]) < 0.1);
  }

  // cuda test
  Tensor cuda_grad_prob = cross_entropy_bwd(
      prob.to(kCUDA), target.to(kCUDA), hice::nullopt, grad_loss.to(kCUDA), 0);
  Tensor cuda_grad_logit =
      softmax_bwd(prob.to(kCUDA), cuda_grad_prob.to(kCUDA), 0).to(kCPU);
  hice::ArrayRef<float> cuda_grad_logit_data(cuda_grad_logit.mutable_data<float>(),
                                        cuda_grad_logit.size());

  Tensor cuda_grad_logit2 =
      softmax_cross_entropy_bwd(prob.to(kCUDA), target.to(kCUDA), hice::nullopt,
                                grad_loss.to(kCUDA), 0)
          .to(kCPU);
  hice::ArrayRef<float> cuda_grad_logit_data2(cuda_grad_logit2.mutable_data<float>(),
                                          cuda_grad_logit2.size());
//   EXPECT_THAT(cuda_grad_logit_data, Pointwise(FloatEq(), cuda_grad_logit_data2));
  for (int i = 0; i < cuda_grad_logit_data.size(); ++ i) {
    EXPECT_TRUE(std::abs(cuda_grad_logit_data[i] - cuda_grad_logit_data2[i]) < 0.1);
  }
}

}  // namespace
}  // namespace hice