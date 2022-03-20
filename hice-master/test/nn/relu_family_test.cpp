#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/nn/activation.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

using scalar_t = double;
#define kDataType kDouble

TEST(ReluFamilyTestFloat, Relu) {
  std::vector<int64_t> dims_in = {12, 34, 56};
  Tensor a_cpu = rand_uniform(dims_in, -10, 10.0, device(kCPU).dtype(kFloat));
  Tensor b_grad_cpu = rand_uniform(dims_in, -10, 10.0, device(kCPU).dtype(kFloat));
  Tensor a_cuda = a_cpu.to(kCUDA);
  Tensor b_grad_cuda = b_grad_cpu.to(kCUDA);
  // fwd
  Tensor b_cpu = relu_fwd(a_cpu);
  Tensor b_cuda = relu_fwd(a_cuda);
  ExpectEqualDenseRegardlessDevice(b_cpu, b_cuda);
  // bwd
  Tensor a_grad_cpu = relu_bwd(a_cpu, b_grad_cpu);
  Tensor a_grad_cuda = relu_bwd(a_cuda, b_grad_cuda);
  ExpectEqualDenseRegardlessDevice(a_grad_cpu, a_grad_cuda);
}

TEST(ReluFamilyTestFloat, Elu) {
  std::vector<int64_t> dims_in = {12};
  Tensor a_cpu = rand_uniform(dims_in, -10, 10.0, device(kCPU).dtype(kFloat));
  Tensor b_grad_cpu = rand_uniform(dims_in, -10, 10.0, device(kCPU).dtype(kFloat));
  Tensor a_cuda = a_cpu.to(kCUDA);
  Tensor b_grad_cuda = b_grad_cpu.to(kCUDA);
  // fwd
  float alpha = 0.5;
  Tensor b_cpu = elu_fwd(a_cpu, alpha);
  Tensor b_cuda = elu_fwd(a_cuda, alpha);
  ExpectEqualDenseRegardlessDevice(b_cpu, b_cuda);
  // bwd
  Tensor a_grad_cpu = elu_bwd(a_cpu, alpha, b_grad_cpu);
  Tensor a_grad_cuda = elu_bwd(a_cuda, alpha, b_grad_cuda);
  ExpectEqualDenseRegardlessDevice(a_grad_cpu, a_grad_cuda);
  // TensorPrinter tp;
  // tp.print(a_cpu);
  // tp.print(b_grad_cpu);
  // tp.print(b_cpu);
  // tp.print(a_grad_cpu);
}

} // namespace hice

