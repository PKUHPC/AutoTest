#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/nn/smooth_l1_loss.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

template <typename TScalarType>
struct SMOOTHL1InplaceTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_target;
  std::vector<int64_t> dims_weight;
  std::vector<int64_t> dims_loss;
};

template <typename TScalarType>
class SMOOTHL1InplaceTest
    : public ::testing::TestWithParam<SMOOTHL1InplaceTestParams<TScalarType>> {};

using SMOOTHL1InplaceTestParamsFloat = SMOOTHL1InplaceTestParams<float>;
using SMOOTHL1InplaceTestFloat = SMOOTHL1InplaceTest<float>;
TEST_P(SMOOTHL1InplaceTestFloat, FWD_None) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  smooth_l1_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::none, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  smooth_l1_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::none, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}

TEST_P(SMOOTHL1InplaceTestFloat, FWD_Mean) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  smooth_l1_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::mean, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform({}, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  smooth_l1_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::mean, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}

TEST_P(SMOOTHL1InplaceTestFloat, FWD_Sum) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  smooth_l1_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::sum, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform({}, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  smooth_l1_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::sum, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}


TEST_P(SMOOTHL1InplaceTestFloat, BWD_None) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  smooth_l1_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::none, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  smooth_l1_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::none, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


TEST_P(SMOOTHL1InplaceTestFloat, BWD_Mean) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  smooth_l1_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::mean, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  smooth_l1_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::mean, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


TEST_P(SMOOTHL1InplaceTestFloat, BWD_Sum) {
  SMOOTHL1InplaceTestParamsFloat params =
      ::testing::TestWithParam<SMOOTHL1InplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  smooth_l1_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::sum, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  smooth_l1_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::sum, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    SMOOTHL1InplaceTestFloatSuite, SMOOTHL1InplaceTestFloat,
    ::testing::Values(
      SMOOTHL1InplaceTestParamsFloat{{5}, 
                                    {5},
                                    {5},
                                    {5}},
      SMOOTHL1InplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2}},
      SMOOTHL1InplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4}, 
                                    {2, 6, 3, 4}, 
                                    {2, 6, 3, 4}},
      SMOOTHL1InplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5}}
    )
);

// #endif
} // namespace hice
 
