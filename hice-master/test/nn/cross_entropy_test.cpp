#include "hice/basic/factories.h"
#include "hice/basic/one_hot.h"
#include "hice/nn/cross_entropy.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ContainerEq;
using ::testing::Pointwise;
using ::testing::FloatEq;

TEST(CrossEntropyForwardTest, CompareCUDAWithCPU) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({8, 10}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor sparse_target =
      rand_uniform({prob.dim(0)}, 0, prob.dim(1), dtype(kInt64).device(kCPU));
  Tensor target = one_hot(sparse_target, prob.dim(1), 1).to(kFloat);

  Tensor cpu_loss = cross_entropy_fwd(prob, target, hice::nullopt, 1);
  hice::ArrayRef<float> cpu_loss_data(cpu_loss.mutable_data<float>(), cpu_loss.size());

  Tensor cuda_loss =
      cross_entropy_fwd(prob.to(kCUDA), target.to(kCUDA), hice::nullopt, 1)
          .to(kCPU);
  hice::ArrayRef<float> cuda_loss_data(cuda_loss.mutable_data<float>(), cuda_loss.size());

  EXPECT_THAT(cuda_loss_data, Pointwise(FloatEq(), cpu_loss_data));

}

TEST(CrossEntropyBackwardTest, CompareCUDAWithCPU) {
  TensorPrinter tp;
  Tensor prob = rand_uniform({10, 12}, 0, 10, dtype(kFloat).device(kCPU));
  Tensor grad_loss =
      rand_uniform({prob.dim(1)}, 0, 4, dtype(kFloat).device(kCPU));

  Tensor sparse_target =
      rand_uniform({prob.dim(1)}, 0, prob.dim(0), dtype(kInt64).device(kCPU));
  Tensor target = one_hot(sparse_target, prob.dim(0), 0).to(kFloat);

  Tensor cpu_grad_prob =
      cross_entropy_bwd(prob, target, hice::nullopt, grad_loss, 0);
  hice::ArrayRef<float> cpu_grad_prob_data(cpu_grad_prob.mutable_data<float>(),
                                       cpu_grad_prob.size());

  Tensor cuda_grad_prob =
      cross_entropy_bwd(prob.to(kCUDA), target.to(kCUDA), hice::nullopt,
                        grad_loss.to(kCUDA), 0).to(kCPU);
  hice::ArrayRef<float> cuda_grad_prob_data(cuda_grad_prob.mutable_data<float>(),
                                       cuda_grad_prob.size());

  EXPECT_THAT(cuda_grad_prob_data, Pointwise(FloatEq(), cpu_grad_prob_data));

  //tp.print(prob);
  //tp.print(grad_loss);
  //tp.print(sparse_target);
  //tp.print(target);
  //tp.print(sparse_grad_prob);
  //tp.print(grad_prob);
}

}
} // namespace hice