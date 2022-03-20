#include <string>
#include <vector>

#include "hice/basic/factories.h"
#include "hice/math/matmul.h"

#include "test/tools/matmul_ref.h"
#include "test/tools/compare.h"
#include "test/tools/timer.h"

#include "test/benchmark/benchmark_cpu.h"
#include "test/benchmark/benchmark_cuda.h"

using namespace hice;

int main(int argc, char** argv) {
  // create Tensor
  Tensor mat1_cpu = rand_normal({1024, 256}, 0, 1, device(kCPU).dtype(kFloat));
  Tensor mat2_cpu = rand_normal({256, 256}, 0, 1, device(kCPU).dtype(kFloat));
  Tensor result_cpu = empty({1024, 256}, mat1_cpu.options());
  Tensor mat1_cuda = mat1_cpu.to(kCUDA);
  Tensor mat2_cuda = mat2_cpu.to(kCUDA);
  Tensor result_cuda = result_cpu.to(kCUDA);

  double time_hice = 0, time_ref = 0;
  double speedup_cpu = 0, speedup_cuda = 0;
  int n_iterations = 100;

  time_ref = BenchmarkCPU::bench(n_iterations, [&](){
    gemm_mklblas(mat1_cpu, mat2_cpu, result_cpu);
  });
  time_hice = BenchmarkCPU::bench(n_iterations, [&](){
    hice::matmul(mat1_cpu, mat2_cpu, result_cpu);
  });
  speedup_cpu = time_ref / time_hice;

  time_ref = BenchmarkCUDA::bench(n_iterations, [&](){
    gemm_cublas(mat1_cuda, mat2_cuda, result_cuda);
  });
  time_hice = BenchmarkCUDA::bench(n_iterations, [&](){
    hice::matmul(mat1_cuda, mat2_cuda, result_cuda);
  });
  speedup_cuda = time_ref / time_hice;
  
  std::cout << "\t[PASSED], CPU Speedup = " <<  speedup_cpu << ", CUDA Speedup = " << speedup_cuda << std::endl;
  std::cout << std::endl;
}