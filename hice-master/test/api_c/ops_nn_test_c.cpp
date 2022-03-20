#include "gtest/gtest.h"

#include "test/tools/compare.h"

extern "C" {
#include "hice/api_c/tensor_impl.h"
#include "hice/hice_c.h"
}

namespace hice {

TEST(NNTestC, Abs) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Abs(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Abs(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Relu) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Relu(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Relu(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Sigmoid) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Sigmoid(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Sigmoid(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Sqrt) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float min = 0, max = 10.0;
  HI_CheckStatus(HI_RandUniform(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &min, &max,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Sqrt(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Sqrt(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Square) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Square(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Square(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Tanh) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Tanh(hi_input_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Tanh(hi_input_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(NNTestC, Elu) {
  // create tensors
  HI_Tensor hi_input_cpu, hi_cpu_output;
  HI_Tensor hi_input_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  float random = -1;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_input_cpu));
  HI_CheckStatus(HI_ToDevice(hi_input_cpu, HI_kCUDA, &hi_input_cuda));
  HI_CheckStatus(HI_Elu(hi_input_cpu, random, &hi_cpu_output));
  HI_CheckStatus(HI_Elu(hi_input_cuda, random, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

}  // namespace hice