#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/reduce.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

extern "C" {
#include "hice/hice_c.h"
#include "hice/api_c/tensor_impl.h"
}

namespace hice {
  template <typename TScalarType>
  struct ReduceTestParams {
    ConstIntArrayRef dims;
    ConstIntArrayRef reduce_dims;
    bool keepdims;
  };

  template <typename TScalarType>
  class ReduceTest
      : public ::testing::TestWithParam<ReduceTestParams<TScalarType>> {};

  using ReduceTestParamsFloat = ReduceTestParams<float>;
  using ReduceTestFloat = ReduceTest<float>;

  TEST_P(ReduceTestFloat, RandUniformValueSumTests) {
    ReduceTestParamsFloat params =
        ::testing::TestWithParam<ReduceTestParamsFloat>::GetParam();
    Tensor cpu_input = rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kDouble)); 
    // Tensor cpu_input = full(params.dims, 1, dtype(kFloat).device(kCPU));  
    Tensor cpu_output = reduce_sum(cpu_input, params.reduce_dims, params.keepdims);
    Tensor cuda_input = cpu_input.to(kCUDA);   
    Tensor cuda_output = reduce_sum(cuda_input, params.reduce_dims, params.keepdims); 
    auto size = cpu_output.size();
    // ExpectEqualDenseWithErrorWithoutType(cpu_output, cuda_output);
    ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output); 
  }

  TEST_P(ReduceTestFloat, RandUniformValueMeanTests) {
    ReduceTestParamsFloat params =
        ::testing::TestWithParam<ReduceTestParamsFloat>::GetParam();
    Tensor cpu_input = rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kDouble));   
    Tensor cpu_output = reduce_mean(cpu_input, params.reduce_dims, params.keepdims);
    Tensor cuda_input = cpu_input.to(kCUDA);   
    Tensor cuda_output = reduce_mean(cuda_input, params.reduce_dims, params.keepdims); 
    auto size = cpu_output.size();
    // ExpectEqualDenseWithErrorWithoutType(cpu_output, cuda_output);
    ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output); 
  }

  TEST_P(ReduceTestFloat, RandUniformValueMaxTests) {
    ReduceTestParamsFloat params =
        ::testing::TestWithParam<ReduceTestParamsFloat>::GetParam();
    Tensor cpu_input = rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));   
    Tensor cpu_output = reduce_max(cpu_input, params.reduce_dims, params.keepdims);   
    Tensor cuda_input = cpu_input.to(kCUDA);   
    Tensor cuda_output = reduce_max(cuda_input, params.reduce_dims, params.keepdims); 
    auto size = cpu_output.size();
    ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
  }

  TEST_P(ReduceTestFloat, RandUniformValueMinTests) {
    ReduceTestParamsFloat params =
        ::testing::TestWithParam<ReduceTestParamsFloat>::GetParam();
    Tensor cpu_input = rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));   
    Tensor cpu_output = reduce_min(cpu_input, params.reduce_dims, params.keepdims);
    Tensor cuda_input = cpu_input.to(kCUDA);   
    Tensor cuda_output = reduce_min(cuda_input, params.reduce_dims, params.keepdims); 
    auto size = cpu_output.size();
    ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
  }

  INSTANTIATE_TEST_CASE_P(
    ReduceTestFloatSuite, ReduceTestFloat,
    ::testing::Values(
      ReduceTestParamsFloat{{3, 2, 2, 3}, {0, 1}, true},
      ReduceTestParamsFloat{{3, 2, 2, 3}, {0, 1}, false},
      ReduceTestParamsFloat{{10, 10, 10, 10}, {0, 3}, true},
      ReduceTestParamsFloat{{10, 10, 10, 10}, {0, 3}, false}
    )
  );
}