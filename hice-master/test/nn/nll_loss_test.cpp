#include "hice/nn/nll_loss.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

template <typename TScalarType>
struct NLLLossInplaceTestParams {
  int64_t batch_size;
  int64_t n_class;
};

template <typename TScalarType>
class NLLLossInplaceTest
    : public ::testing::TestWithParam<NLLLossInplaceTestParams<TScalarType>> {};

using NLLLossInplaceTestParamsFloat = NLLLossInplaceTestParams<float>;
using NLLLossInplaceTestFloat = NLLLossInplaceTest<float>;

TEST_P(NLLLossInplaceTestFloat, FWD) {
  NLLLossInplaceTestParamsFloat params =
      ::testing::TestWithParam<NLLLossInplaceTestParamsFloat>::GetParam();
  int64_t batch_size = params.batch_size;
  int64_t n_class = params.n_class;

  Tensor cpu_input = rand_uniform({batch_size, n_class}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform({batch_size}, 0, n_class, device(kCPU).dtype(kInt64));
  Tensor cpu_weight = rand_uniform({n_class}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_loss = rand_uniform({batch_size}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  nll_loss_fwd(cpu_input, cpu_target, cpu_weight, cpu_loss);

  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform({batch_size}, 0, 10, device(kCUDA).dtype(kFloat));
  nll_loss_fwd(cuda_input, cuda_target, cuda_weight, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
  // TensorPrinter tp;
  // tp.print(cpu_input);
  // tp.print(cpu_target);
  // tp.print(cpu_weight);
  // tp.print(cpu_loss);
  // tp.print(cuda_loss);
}


TEST_P(NLLLossInplaceTestFloat, BWD) {
  NLLLossInplaceTestParamsFloat params =
      ::testing::TestWithParam<NLLLossInplaceTestParamsFloat>::GetParam();
  int64_t batch_size = params.batch_size;
  int64_t n_class = params.n_class;

  Tensor cpu_input = rand_uniform({batch_size, n_class}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform({batch_size}, 0, n_class, device(kCPU).dtype(kInt64));
  Tensor cpu_weight = rand_uniform({n_class}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_loss = rand_uniform({batch_size}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_input = rand_uniform({batch_size, n_class}, 0.0, 10.0, device(kCPU).dtype(kFloat));
  nll_loss_bwd(cpu_input, cpu_target, cpu_weight, cpu_grad_loss, cpu_grad_input);

  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform({batch_size, n_class}, 0, 10, device(kCUDA).dtype(kFloat));
  nll_loss_bwd(cuda_input, cuda_target, cuda_weight, cuda_grad_loss, cuda_grad_input);
  // dims compare(cpu, cuda, Truth)
  EXPECT_EQ(cpu_grad_input.ndim(), cuda_grad_input.ndim());
  for (size_t i = 0; i < cpu_grad_input.ndim(); ++i) {
    EXPECT_EQ(cpu_grad_input.dim(i), cuda_grad_input.dim(i));
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_grad_input.size();
  auto size_cuda_output = cuda_grad_input.size();
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  Tensor cuda_output_host = cuda_grad_input.to(kCPU);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_FLOAT_EQ(cpu_grad_input.data<float>()[i],
                    cuda_output_host.data<float>()[i]);
  }
  // TensorPrinter tp;
  // tp.print(cpu_input);
  // tp.print(cpu_target);
  // tp.print(cpu_weight);
  // tp.print(cpu_grad_loss);
  // tp.print(cpu_grad_input);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    NLLLossInplaceTestFloatSuite, NLLLossInplaceTestFloat,
    ::testing::Values(
      NLLLossInplaceTestParamsFloat{4, 5},
      NLLLossInplaceTestParamsFloat{6, 3000},
      NLLLossInplaceTestParamsFloat{10000, 200}
    )
);

// #endif
}  // namespace hice
