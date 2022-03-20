#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/nn/activation.h"
#include <sys/time.h>
#include "gtest/gtest.h"

namespace hice {

// #if 0
// unary
using ActivationFWDOp = std::function<Tensor(Tensor& a)>;
auto ABS_FWD  = [](Tensor& a)->Tensor{ return abs_fwd(a); };
auto RELU_FWD  = [](Tensor& a)->Tensor{ return relu_fwd(a); };
auto SIGMOID_FWD  = [](Tensor& a)->Tensor{ return sigmoid_fwd(a); };
auto SQRT_FWD  = [](Tensor& a)->Tensor{ return sqrt_fwd(a); };
auto SQUARE_FWD  = [](Tensor& a)->Tensor{ return square_fwd(a); };
auto TANH_FWD  = [](Tensor& a)->Tensor{ return tanh_fwd(a); };

template <typename TScalarType>
struct ActivationFWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_output;
  ActivationFWDOp Op;
};

template <typename TScalarType>
class ActivationFWDTest
    : public ::testing::TestWithParam<ActivationFWDTestParams<TScalarType>> {};

using ActivationFWDTestParamsFloat = ActivationFWDTestParams<float>;
using ActivationFWDTestFloat = ActivationFWDTest<float>;
TEST_P(ActivationFWDTestFloat, RandUniformValueTests) {
  ActivationFWDTestParamsFloat params =
      ::testing::TestWithParam<ActivationFWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  struct timeval hice_start, hice_end;
  double hice_time;
  gettimeofday(&hice_start,NULL);

  Tensor cpu_output = params.Op(cpu_input);
  gettimeofday(&hice_end,NULL);
    hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
              + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


  std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;

//  Tensor cuda_input = cpu_input.to(kCUDA);
//  Tensor cuda_output = params.Op(cuda_input);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_output.size();
  EXPECT_EQ(cpu_output.ndim(), ndim_output_truth);
//  EXPECT_EQ(cuda_output.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_output.dim(i), params.dims_output[i]);
//    EXPECT_EQ(cuda_output.dim(i), params.dims_output[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_output.size();
//  std:printf()
//  auto size_cuda_output = cuda_output.size();
//  Tensor cuda_output_host = cuda_output.to(kCPU);
//  EXPECT_EQ(size_cpu_output, size_cuda_output);
//  for(int i = 0; i < size_cpu_output; ++i) {
//    EXPECT_LE(std::abs(cpu_output.data<float>()[i]),
//              1e-5);
//  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    ActivationFloatSuite, ActivationFWDTestFloat,
    ::testing::Values(
      // scalar
      ActivationFWDTestParamsFloat{{}, 
                                  {},
                                  ABS_FWD},
      // tensor
      ActivationFWDTestParamsFloat{{50000}, 
                                  {50000},
                                  ABS_FWD},
      ActivationFWDTestParamsFloat{{100000}, 
                                  {100000},
                                  RELU_FWD},
      ActivationFWDTestParamsFloat{{800000}, 
                                  {800000},
                                  SIGMOID_FWD},
      ActivationFWDTestParamsFloat{{500000}, 
                                  {500000},
                                  SQRT_FWD},
      ActivationFWDTestParamsFloat{{500000}, 
                                  {500000},
                                  SQUARE_FWD},
      // parallel(size>=2^15)
      ActivationFWDTestParamsFloat{{350000}, 
                                  {350000},
                                  TANH_FWD}
    )
);


// #endif


using ActivationBWDOp = std::function<Tensor(Tensor& a, Tensor& b)>;
auto ABS_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return abs_bwd(a, b); };
auto RELU_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return relu_bwd(a, b); };
auto SIGMOID_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return sigmoid_bwd(a, b); };
auto SQRT_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return sqrt_bwd(a, b); };
auto SQUARE_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return square_bwd(a, b); };
auto TANH_BWD  = [](Tensor& a, Tensor& b)->Tensor{ return tanh_bwd(a, b); };

template <typename TScalarType>
struct ActivationBWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_grad_output;
  std::vector<int64_t> dims_grad_input;
  ActivationBWDOp Op;
};

template <typename TScalarType>
class ActivationBWDTest
    : public ::testing::TestWithParam<ActivationBWDTestParams<TScalarType>> {};

using ActivationBWDTestParamsFloat = ActivationBWDTestParams<float>;
using ActivationBWDTestFloat = ActivationBWDTest<float>;
TEST_P(ActivationBWDTestFloat, RandUniformValueTests) {
  ActivationBWDTestParamsFloat params =
      ::testing::TestWithParam<ActivationBWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_grad_output = rand_uniform(params.dims_grad_output, 1.0, 10.0, device(kCPU).dtype(kFloat));   
//  Tensor cuda_input = cpu_input.to(kCUDA);
//  Tensor cuda_grad_output = cpu_grad_output.to(kCUDA);
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);
  Tensor cpu_grad_input = params.Op(cpu_input, cpu_grad_output);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;

//  Tensor cuda_grad_input = params.Op(cuda_input, cuda_grad_output);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_grad_input.size();
  EXPECT_EQ(cpu_grad_input.ndim(), ndim_output_truth);
//  EXPECT_EQ(cuda_grad_input.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_grad_input.dim(i), params.dims_grad_input[i]);
//    EXPECT_EQ(cuda_grad_input.dim(i), params.dims_grad_input[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_grad_input.size();
//  auto size_cuda_output = cuda_grad_input.size();
//  Tensor cuda_output_host = cuda_grad_input.to(kCPU);
//  EXPECT_EQ(size_cpu_output, size_cuda_output);
//  for(int i = 0; i < size_cpu_output; ++i) {
//    EXPECT_LE(std::abs(cpu_grad_input.data<float>()[i] - cuda_output_host.data<float>()[i]),
//              1e-5);
//  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    ActivationFloatSuite, ActivationBWDTestFloat,
    ::testing::Values(
      // scalar
      ActivationBWDTestParamsFloat{{}, 
                                  {},
                                  {},
                                  ABS_BWD},
      // tensor
      ActivationBWDTestParamsFloat{{6}, 
                                  {6},
                                  {6},
                                  ABS_BWD},
      ActivationBWDTestParamsFloat{{50000}, 
                                  {50000},
                                  {50000},
                                  RELU_BWD},
      ActivationBWDTestParamsFloat{{200000}, 
                                  {200000},
                                  {200000},
                                  SIGMOID_BWD},
      ActivationBWDTestParamsFloat{{100000}, 
                                  {100000},
                                  {100000},
                                  SQRT_BWD},
      ActivationBWDTestParamsFloat{{1000000}, 
                                  {1000000},
                                  {1000000},
                                  SQUARE_BWD},
      // parallel(size>=2^15)
      ActivationBWDTestParamsFloat{{100000}, 
                                  {100000},
                                  {100000},
                                  TANH_BWD}
    )
);


} // namespace anonymous
