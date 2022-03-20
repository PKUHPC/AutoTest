#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/nn/softmax.h"

#include <vector>
#include "gtest/gtest.h"

namespace hice {

template <typename TScalarType>
struct SoftmaxFwdTestParams {
  std::vector<int64_t> dims;
  int axis;
  TScalarType fill_value;
};

template <typename TScalarType>
class SoftmaxFwdTest
    : public ::testing::TestWithParam<SoftmaxFwdTestParams<TScalarType>> {};

using SoftmaxFwdTestParamsFloat = SoftmaxFwdTestParams<float>;
using SoftmaxFwdTestFloat = SoftmaxFwdTest<float>;

TEST_P(SoftmaxFwdTestFloat, FullSameValueTests) {
  SoftmaxFwdTestParamsFloat params =
      ::testing::TestWithParam<SoftmaxFwdTestParamsFloat>::GetParam();
  Tensor cpu_input =
      full(params.dims, params.fill_value, device(kCPU).dtype(kFloat));
  Tensor cpu_output = softmax_fwd(cpu_input, params.axis);
  Tensor cuda_input =
      full(params.dims, params.fill_value, device(kCUDA).dtype(kFloat));
  Tensor cuda_output = softmax_fwd(cuda_input, params.axis);
  Tensor cuda_output_host = cuda_output.to(kCPU);
  auto size = cpu_input.size();
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(cpu_output.data<float>()[i],
                    cuda_output_host.data<float>()[i]);
  }
}

TEST_P(SoftmaxFwdTestFloat, RandUniformValueTests) {
  SoftmaxFwdTestParamsFloat params =
      ::testing::TestWithParam<SoftmaxFwdTestParamsFloat>::GetParam();
  Tensor cpu_input =
      rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_output = softmax_fwd(cpu_input, params.axis);
  TensorPrinter tp;
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_output = softmax_fwd(cpu_input, params.axis);
  Tensor cuda_output_host = cuda_output.to(kCPU);
  auto size = cpu_input.size();
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(cpu_output.data<float>()[i],
                    cuda_output_host.data<float>()[i]);
  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    SoftmaxFwdTestFloatSuite, SoftmaxFwdTestFloat,
    ::testing::Values(SoftmaxFwdTestParamsFloat{{1}, 0, 1.0},
                      SoftmaxFwdTestParamsFloat{{2, 2}, 0, 2.0},
                      SoftmaxFwdTestParamsFloat{{2, 2}, 1, 2.0},
                      SoftmaxFwdTestParamsFloat{{3, 3, 3}, 0, 3.0},
                      SoftmaxFwdTestParamsFloat{{3, 3, 3}, 1, 3.0},
                      SoftmaxFwdTestParamsFloat{{3, 3, 3}, 2, 3.0},
                      SoftmaxFwdTestParamsFloat{{4, 4, 4, 4}, 0, 4.0},
                      SoftmaxFwdTestParamsFloat{{4, 4, 4, 4}, 1, 4.0},
                      SoftmaxFwdTestParamsFloat{{4, 4, 4, 4}, 2, 4.0},
                      SoftmaxFwdTestParamsFloat{{4, 4, 4, 4}, 3, 4.0}));

template <typename TScalarType>
struct SoftmaxBwdTestParams {
  std::vector<int64_t> dims;
  int axis;
  TScalarType fill_value;
};

template <typename TScalarType>
class SoftmaxBwdTest
    : public ::testing::TestWithParam<SoftmaxBwdTestParams<TScalarType>> {};

using SoftmaxBwdTestParamsFloat = SoftmaxBwdTestParams<float>;
using SoftmaxBwdTestFloat = SoftmaxBwdTest<float>;

TEST_P(SoftmaxBwdTestFloat, FullSameValueTests) {
  SoftmaxBwdTestParamsFloat params =
      ::testing::TestWithParam<SoftmaxBwdTestParamsFloat>::GetParam();
  Tensor cpu_output = full(params.dims, 1, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_output = full(params.dims, 2, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_input = softmax_bwd(cpu_output, cpu_grad_output, params.axis);
  Tensor cuda_output = full(params.dims, 1, device(kCUDA).dtype(kFloat));
  Tensor cuda_grad_output = full(params.dims, 2, device(kCUDA).dtype(kFloat));
  Tensor cuda_grad_input =
      softmax_bwd(cuda_output, cuda_grad_output, params.axis);
  Tensor cuda_grad_input_host = cuda_grad_input.to(kCPU);
  auto size = cpu_grad_input.size();
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(cpu_grad_input.data<float>()[i],
                    cuda_grad_input_host.data<float>()[i]);
  }
}

TEST_P(SoftmaxBwdTestFloat, RandUniformValueTests) {
  SoftmaxBwdTestParamsFloat params =
      ::testing::TestWithParam<SoftmaxBwdTestParamsFloat>::GetParam();
  Tensor cpu_output =
      rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_output =
      rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_grad_input = softmax_bwd(cpu_output, cpu_grad_output, params.axis);
  Tensor cuda_output = cpu_output.to(kCUDA);
  Tensor cuda_grad_output = cpu_grad_output.to(kCUDA);
  Tensor cuda_grad_input =
      softmax_bwd(cuda_output, cuda_grad_output, params.axis);
  Tensor cuda_grad_input_host = cuda_grad_input.to(kCPU);
  auto size = cpu_grad_input.size();
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(cpu_grad_input.data<float>()[i],
                    cuda_grad_input_host.data<float>()[i]);
  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    SoftmaxBwdTestFloatSuite, SoftmaxBwdTestFloat,
    ::testing::Values(SoftmaxBwdTestParamsFloat{{1}, 0, 1.0},
                      SoftmaxBwdTestParamsFloat{{2, 2}, 0, 2.0},
                      SoftmaxBwdTestParamsFloat{{2, 2}, 1, 2.0},
                      SoftmaxBwdTestParamsFloat{{3, 3, 3}, 0, 3.0},
                      SoftmaxBwdTestParamsFloat{{3, 3, 3}, 1, 3.0},
                      SoftmaxBwdTestParamsFloat{{3, 3, 3}, 2, 3.0},
                      SoftmaxBwdTestParamsFloat{{4, 4, 4, 4}, 0, 4.0},
                      SoftmaxBwdTestParamsFloat{{4, 4, 4, 4}, 1, 4.0},
                      SoftmaxBwdTestParamsFloat{{4, 4, 4, 4}, 2, 4.0},
                      SoftmaxBwdTestParamsFloat{{4, 4, 4, 4}, 3, 4.0}));

}  // namespace hice
