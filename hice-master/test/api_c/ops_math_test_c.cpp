#include "gtest/gtest.h"

#include "test/tools/compare.h"

#include "hice/math/matmul.h"

extern "C" {
#include "hice/api_c/tensor_impl.h"
#include "hice/hice_c.h"
}

namespace hice {

TEST(MathTestC, Matmul) {
  // create tensors
  HI_Tensor hi_tensor1_cpu, hi_tensor2_cpu, hi_cpu_output;
  HI_Tensor hi_tensor1_cuda, hi_tensor2_cuda, hi_cuda_output;
  std::vector<int64_t> dims = {2, 2};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_tensor1_cpu));
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_tensor2_cpu));
  HI_CheckStatus(HI_ToDevice(hi_tensor1_cpu, HI_kCUDA, &hi_tensor1_cuda));
  HI_CheckStatus(HI_ToDevice(hi_tensor2_cpu, HI_kCUDA, &hi_tensor2_cuda));
  HI_CheckStatus(HI_Matmul(hi_tensor1_cpu, hi_tensor2_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Matmul(hi_tensor1_cuda, hi_tensor2_cuda, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(MathTestC, Reduce) {
  // create tensors
  HI_Tensor hi_tensor_cpu, hi_tensor_cuda, hi_cpu_output, hi_cuda_output;
  std::vector<int64_t> dims = {3, 2, 2, 3};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_tensor_cpu));
  HI_CheckStatus(HI_ToDevice(hi_tensor_cpu, HI_kCUDA, &hi_tensor_cuda));
  // params
  HI_ReduceMode mode = SUM;
  bool keep_dim = false;
  std::vector<int64_t> axises = {0, 1};
  // reduce
  HI_CheckStatus(HI_Reduce(hi_tensor_cpu, mode, axises.data(), axises.size(),
                           keep_dim, &hi_cpu_output));
  HI_CheckStatus(HI_Reduce(hi_tensor_cuda, mode, axises.data(), axises.size(),
                           keep_dim, &hi_cuda_output));
  // compare
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(MathTestC, Binary) {
  // create tensors
  HI_Tensor hi_tensor1_cpu, hi_tensor1_cuda, hi_tensor2_cpu, hi_tensor2_cuda,
      hi_cpu_output, hi_cuda_output;
  std::vector<int64_t> dims = {3, 2, 2, 3};
  float mean = 0, stddev = 10.0;
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_tensor1_cpu));
  HI_CheckStatus(HI_RandNormal(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &mean, &stddev,
                         &hi_tensor2_cpu));

  HI_CheckStatus(HI_ToDevice(hi_tensor1_cpu, HI_kCUDA, &hi_tensor1_cuda));
  HI_CheckStatus(HI_ToDevice(hi_tensor2_cpu, HI_kCUDA, &hi_tensor2_cuda));
  // binary add
  HI_CheckStatus(HI_Add(hi_tensor1_cpu, hi_tensor2_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Add(hi_tensor1_cuda, hi_tensor2_cuda, &hi_cuda_output));

  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(MathTestC, Unary) {
  // create tensors
  HI_Tensor hi_tensor_cpu, hi_tensor_cuda, hi_cpu_output, hi_cuda_output;
  std::vector<int64_t> dims = {3, 2, 2, 3};
  float min = 0, max = 10.0;
  HI_CheckStatus(HI_RandUniform(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &min, &max,
                         &hi_tensor_cpu));

  HI_CheckStatus(HI_ToDevice(hi_tensor_cpu, HI_kCUDA, &hi_tensor_cuda));
  // unary exp
  HI_CheckStatus(HI_Exp(hi_tensor_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Exp(hi_tensor_cuda, &hi_cuda_output));

  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
  // unary log
  HI_CheckStatus(HI_Log(hi_tensor_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Log(hi_tensor_cuda, &hi_cuda_output));
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
  // unary neg
  HI_CheckStatus(HI_Neg(hi_tensor_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Neg(hi_tensor_cuda, &hi_cuda_output));
  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}

TEST(MathTestC, Compare) {
  // create tensors
  HI_Tensor hi_tensor1_cpu, hi_tensor1_cuda, hi_tensor2_cpu, hi_tensor2_cuda,
      hi_cpu_output, hi_cuda_output;
  std::vector<int64_t> dims = {3, 2, 2, 3};
  float a = 0, b = 10;
  HI_CheckStatus(HI_RandUniform(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &a, &b,
                         &hi_tensor1_cpu));
  HI_CheckStatus(HI_RandUniform(HI_kFloat, HI_kCPU, dims.data(), dims.size(), &a, &b,
                         &hi_tensor2_cpu));

  HI_CheckStatus(HI_ToDevice(hi_tensor1_cpu, HI_kCUDA, &hi_tensor1_cuda));
  HI_CheckStatus(HI_ToDevice(hi_tensor2_cpu, HI_kCUDA, &hi_tensor2_cuda));
  // compare equal 
  HI_CheckStatus(HI_Equal(hi_tensor1_cpu, hi_tensor2_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Equal(hi_tensor1_cuda, hi_tensor2_cuda, &hi_cuda_output));

  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);

  // compare less 
  HI_CheckStatus(HI_Less(hi_tensor1_cpu, hi_tensor2_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Less(hi_tensor1_cuda, hi_tensor2_cuda, &hi_cuda_output));

  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);

  // compare greater 
  HI_CheckStatus(HI_Greater(hi_tensor1_cpu, hi_tensor2_cpu, &hi_cpu_output));
  HI_CheckStatus(HI_Greater(hi_tensor1_cuda, hi_tensor2_cuda, &hi_cuda_output));

  ExpectEqualDenseRegardlessDevice(hi_cpu_output->tensor_,
                                   hi_cuda_output->tensor_);
}


}  // namespace hice