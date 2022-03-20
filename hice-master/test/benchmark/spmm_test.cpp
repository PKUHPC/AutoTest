#include <string>
#include <vector>

#include "hice/basic/factories.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/matmul.h"

#include "test/tools/matrix.h"
#include "test/tools/matmul_ref.h"
#include "test/tools/timer.h"
#include "test/tools/compare.h"

using namespace hice;

int main(int argc, char** argv) {
  int k = 256;
  std::string file_name;
  if (argc == 3) {
    file_name = argv[1];
    k = atoi(argv[2]);
  } else {
    std::cout << "Usage: matrix_file_name, k(the height of second matrix)." << std::endl;
    exit(1);
  }

  /**************read sparse matrix from file.*******************/
  MTX<double> mtx;
  fileToMtxCoo<double>(file_name.c_str(), &mtx, true); 
  int m = mtx.rows;
  int n = mtx.cols;
  int mat_nnz = mtx.nnz;
  std::cout << "spmm_test" << k << ":" << std::endl;
  std::cout << "\t" << file_name << ",  m=" << m << ", n=" << n
            << ", nnz=" << mat_nnz << std::endl;
  std::vector<double> values(mtx.data, mtx.data + mat_nnz);
  std::vector<int32_t> indices(mat_nnz * 2, 0);
  for (int i = 0; i < mat_nnz; ++i) {
    indices[i] = mtx.row[i];
    indices[i + mat_nnz] = mtx.col[i];
  }
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);

  // create Tensor on cpu
  Tensor mat1_sparse_cpu = wrap_sparse({m, n}, indices.data(), values.data(), mat_nnz, device(kCPU).dtype(kDouble));
  Tensor mat2_dense_cpu = full({n, k}, 1, device(kCPU).dtype(kDouble));
  Tensor mat1_dense_cpu = mat1_sparse_cpu.to(kDense);
  // create Tensor on cuda
  Tensor mat1_sparse_cuda = mat1_sparse_cpu.to(kCUDA);
  Tensor mat2_dense_cuda = mat2_dense_cpu.to(kCUDA);
  Tensor mat1_dense_cuda = mat1_dense_cpu.to(kCUDA);
  mat1_sparse_cuda.set_coalesced(true);

  double time_ge = 0, time_sp = 0;
  Tensor result_spmm, result_gemm;

/////////CPU TEST//////////
{
  // std::cout << "CPU SPMM: ";
  timeval t1, t2;
  gettimeofday(&t1,NULL); 
  for (int i = 0; i < NUM_RUN; i++) {
    result_spmm = hice::matmul(mat1_sparse_cpu, mat2_dense_cpu);
  }
  gettimeofday(&t2,NULL); 
  time_sp = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / NUM_RUN;
}
{
  // std::cout << "CPU GEMM: ";
  timeval t1, t2;
  gettimeofday(&t1,NULL); 
  for (int i = 0; i < NUM_RUN; i++) {
    result_gemm = gemm_mklblas(mat1_dense_cpu, mat2_dense_cpu);
  }
  gettimeofday(&t2,NULL); 
  time_ge = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / NUM_RUN;
}
  ExpectEqualDenseWithError(result_spmm, result_gemm);
  double speedup_cpu = time_ge / time_sp;

/////////GPU TEST//////////
{
  // Warm-up calling before timing.
  Tensor result = hice::matmul(mat1_sparse_cuda, mat2_dense_cuda);
  Tensor result1 = hice::matmul(mat1_dense_cuda, mat2_dense_cuda);
}
{
  // std::cout << "CUDA SPMM: ";
  cuda_timer timer;
  timer.start();
  for (int i = 0; i < NUM_RUN; i++) {
    result_spmm = hice::matmul(mat1_sparse_cuda, mat2_dense_cuda);
  }
  time_sp = timer.stop() * 1.0 / NUM_RUN;
}
{
  // std::cout << "CUDA GEMM: ";
  cuda_timer timer;
  timer.start();
  for (int i = 0; i < NUM_RUN; i++) {
    result_gemm = gemm_cublas(mat1_dense_cuda, mat2_dense_cuda);
  }
  time_ge = timer.stop() * 1.0 / NUM_RUN;
}
  ExpectEqualDenseWithError(result_spmm, result_gemm);
  double speedup_gpu = time_ge / time_sp;

  std::cout << "\t[PASSED], CPU Speedup = " <<  speedup_cpu 
            << ", GPU Speedup = " << speedup_gpu << std::endl;
  std::cout << std::endl;
}