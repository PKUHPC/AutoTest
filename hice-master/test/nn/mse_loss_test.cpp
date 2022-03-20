#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/nn/mse_loss.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

template <typename TScalarType>
struct MSELossInplaceTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_target;
  std::vector<int64_t> dims_weight;
  std::vector<int64_t> dims_loss;
};

template <typename TScalarType>
class MSELossInplaceTest
    : public ::testing::TestWithParam<MSELossInplaceTestParams<TScalarType>> {};

using MSELossInplaceTestParamsFloat = MSELossInplaceTestParams<float>;
using MSELossInplaceTestFloat = MSELossInplaceTest<float>;
TEST_P(MSELossInplaceTestFloat, FWD_None) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  mse_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::none, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  mse_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::none, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}

TEST_P(MSELossInplaceTestFloat, FWD_Mean) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  mse_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::mean, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform({}, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  mse_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::mean, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}

TEST_P(MSELossInplaceTestFloat, FWD_Sum) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  mse_loss_fwd(cpu_input, cpu_target, cpu_weight, Reduction::sum, cpu_loss);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_loss = rand_uniform({}, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  mse_loss_fwd(cuda_input, cuda_target, cuda_weight, Reduction::sum, cuda_loss);
  ExpectEqualDenseRegardlessDevice(cpu_loss, cuda_loss);
}


TEST_P(MSELossInplaceTestFloat, BWD_None) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform(params.dims_loss, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  mse_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::none, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  mse_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::none, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


TEST_P(MSELossInplaceTestFloat, BWD_Mean) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  mse_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::mean, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  mse_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::mean, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


TEST_P(MSELossInplaceTestFloat, BWD_Sum) {
  MSELossInplaceTestParamsFloat params =
      ::testing::TestWithParam<MSELossInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_target = rand_uniform(params.dims_target, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_weight = rand_uniform(params.dims_weight, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_loss = rand_uniform({}, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  Tensor cpu_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  mse_loss_bwd(cpu_input, cpu_target, cpu_weight, Reduction::sum, cpu_grad_loss, cpu_grad_input);
  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_weight = cpu_weight.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  Tensor cuda_grad_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCUDA).dtype(kFloat)); 
  mse_loss_bwd(cuda_input, cuda_target, cuda_weight, Reduction::sum, cuda_grad_loss, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    MSELossInplaceTestFloatSuite, MSELossInplaceTestFloat,
    ::testing::Values(
      MSELossInplaceTestParamsFloat{{5}, 
                                    {5},
                                    {5},
                                    {5}},
      MSELossInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2}},
      MSELossInplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4}, 
                                    {2, 6, 3, 4}, 
                                    {2, 6, 3, 4}},
      MSELossInplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5}}
    )
);

// #endif
} // namespace hice
 
