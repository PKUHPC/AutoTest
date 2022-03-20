#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/arg_reduce.h"

#include "gtest/gtest.h"
#include "test/tools/compare.h"

namespace hice {
  template <typename TScalarType>
  struct ArgReduceTestParams {
    ConstIntArrayRef dims;
    int64_t reduce_dims;
    bool keepdims;
  };

  template <typename TScalarType>
  class ArgReduceTest
      : public ::testing::TestWithParam<ArgReduceTestParams<TScalarType>> {};

  using ArgReduceTestParamsFloat = ArgReduceTestParams<float>;
  using ArgReduceTestFloat = ArgReduceTest<float>;

  TEST_P(ArgReduceTestFloat, RandUniformValueMinTests) {
    ArgReduceTestParamsFloat params =
        ::testing::TestWithParam<ArgReduceTestParamsFloat>::GetParam();
    Tensor cpu_input = rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));   
    auto cpu_output = min(cpu_input, params.reduce_dims, params.keepdims);   
    Tensor min_data_cpu = std::get<0>(cpu_output);
    Tensor min_indice_cpu = std::get<1>(cpu_output);
    // TensorPrinter tp;
    // tp.print(cpu_input);
    // tp.print(min_data_cpu);
    // tp.print(min_indice_cpu);
    Tensor cuda_input = cpu_input.to(kCUDA);
    auto cuda_output = min(cuda_input, params.reduce_dims, params.keepdims);
    Tensor min_data_cuda = std::get<0>(cuda_output).to(kCPU);
    Tensor min_indice_cuda = std::get<1>(cuda_output).to(kCPU);
    ExpectEqualDenseRegardlessDevice(min_data_cpu, min_data_cuda);
    ExpectEqualDenseRegardlessDevice(min_indice_cpu, min_indice_cuda);
  }

  TEST_P(ArgReduceTestFloat, RandUniformValueMaxTests) {
    ArgReduceTestParamsFloat params =
        ::testing::TestWithParam<ArgReduceTestParamsFloat>::GetParam();
    Tensor cpu_input =
        rand_uniform(params.dims, 1.0, 10.0, device(kCPU).dtype(kFloat));
    auto cpu_output = max(cpu_input, params.reduce_dims, params.keepdims);
    Tensor max_data_cpu = std::get<0>(cpu_output);
    Tensor max_indice_cpu = std::get<1>(cpu_output);
    // TensorPrinter tp;
    // tp.print(cpu_input);
    // tp.print(max_data);
    // tp.print(max_indice);
    Tensor cuda_input = cpu_input.to(kCUDA);
    auto cuda_output = max(cuda_input, params.reduce_dims, params.keepdims);
    Tensor max_data_cuda = std::get<0>(cuda_output).to(kCPU);
    Tensor max_indice_cuda = std::get<1>(cuda_output).to(kCPU);
    ExpectEqualDenseRegardlessDevice(max_data_cpu, max_data_cuda);
    ExpectEqualDenseRegardlessDevice(max_indice_cpu, max_indice_cuda);
  }

  INSTANTIATE_TEST_CASE_P(
    ArgReduceTestFloatSuite, ArgReduceTestFloat,
    ::testing::Values(
      ArgReduceTestParamsFloat{{3, 2}, 1, true},
      ArgReduceTestParamsFloat{{3, 2}, 1, false},
      ArgReduceTestParamsFloat{{3, 2, 2, 3}, 1, true},
      ArgReduceTestParamsFloat{{3, 2, 2, 3}, 0, true},
      ArgReduceTestParamsFloat{{3, 2, 2, 3}, 0, false}
    )
  );
}