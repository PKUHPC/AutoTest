#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/nn/dropout.h"

#include "gtest/gtest.h"

namespace hice{
  template <typename TScalarType>
  struct DropoutTestParams {
    ConstIntArrayRef dims;
    double rate;
    float initial_value;
  };

  template <typename TScalarType>
  class DropoutTest
      : public ::testing::TestWithParam<DropoutTestParams<TScalarType>> {};

  using DropoutTestParamsFloat = DropoutTestParams<float>;
  using DropoutTestFloat = DropoutTest<float>;

  TEST_P(DropoutTestFloat, FullValueDropoutForwardTests) {
    DropoutTestParamsFloat params =
        ::testing::TestWithParam<DropoutTestParamsFloat>::GetParam();
    Tensor cpu_input = full(params.dims, params.initial_value, device(kCPU).dtype(kFloat));
    Tensor cpu_mask(cpu_input.dims(),
                device(cpu_input.device()).dtype(kBool).layout(kDense));
    Tensor cpu_output = dropout_fwd(cpu_input, params.rate, cpu_mask);
    Tensor cuda_input = cpu_input.to(kCUDA);
    Tensor cuda_mask = cpu_mask.to(kCUDA);
    Tensor cuda_output = dropout_fwd(cuda_input, params.rate, cuda_mask); 
    Tensor cuda_output_host = cuda_output.to(kCPU);
    auto size = cpu_output.size();
    int cpu_count = 0, cuda_count = 0;
    for(int i = 0; i < size; ++i) {
      if(cpu_output.data<float>()[i] == 0){
        cpu_count++;
      }
      if(cuda_output_host.data<float>()[i] == 0){
        cuda_count++;
      }
    }
    EXPECT_FLOAT_EQ(cpu_count/size, cuda_count/size);
  }

  TEST_P(DropoutTestFloat, FullValueDropoutBackwardTests) {
    DropoutTestParamsFloat params =
        ::testing::TestWithParam<DropoutTestParamsFloat>::GetParam();
    Tensor cpu_input = full(params.dims, params.initial_value, device(kCPU).dtype(kFloat));
    Tensor cpu_mask(cpu_input.dims(),
                device(cpu_input.device()).dtype(kBool).layout(kDense));
    Tensor cpu_output = dropout_fwd(cpu_input, params.rate, cpu_mask);
    Tensor cuda_input = cpu_input.to(kCUDA);
    Tensor cuda_mask = cpu_mask.to(kCUDA);
    Tensor cuda_output = dropout_fwd(cuda_input, params.rate, cuda_mask); 

    Tensor backward_cpu_output =
        dropout_bwd(cpu_input, params.rate, cpu_mask);
    Tensor backward_cuda_output =
        dropout_bwd(cuda_input, params.rate, cuda_mask);
    Tensor backward_cuda_output_host = backward_cuda_output.to(kCPU);
    auto size = cpu_output.size();
    int cpu_count = 0, cuda_count = 0;
    for(int i = 0; i < size; ++i) {
      if(backward_cpu_output.data<float>()[i] == 0){
        cpu_count++;
      }
      if(backward_cuda_output_host.data<float>()[i] == 0){
        cuda_count++;
      }
    }
    // std::cout << cpu_count << " " << cuda_count << std::endl;
    EXPECT_FLOAT_EQ(cpu_count/size, cuda_count/size);
  }

  INSTANTIATE_TEST_CASE_P(
    DropoutTestFloatSuite, DropoutTestFloat,
    ::testing::Values(
      DropoutTestParamsFloat{{5, 5, 5, 5}, 0.5, 5},
      DropoutTestParamsFloat{{5, 5, 5, 5}, 0.7, 5},
      DropoutTestParamsFloat{{100, 100}, 0.5, 5},
      DropoutTestParamsFloat{{100, 100}, 0.7, 5}
    )
  );
}