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
  int k = 1;
  std::string file_name;
  if (argc == 2) {
    file_name = argv[1];
  } else {
    std::cout << "Usage: matrix_file_name." << std::endl;
    exit(1);
  }

  /**************read sparse matrix from file.*******************/
  MTX<double> mtx;
  fileToMtxCoo<double>(file_name.c_str(), &mtx, true); 
  int m = mtx.rows;
  int n = mtx.cols;
  int mat_nnz = mtx.nnz;
  std::cout << "spmv_test:" << std::endl;
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
  Tensor mat_sparse_cpu = wrap_sparse({m, n}, indices.data(), values.data(), mat_nnz, device(kCPU).dtype(kDouble));
  Tensor vec_dense_cpu = full({n}, 1, device(kCPU).dtype(kDouble));
  Tensor mat_dense_cpu = mat_sparse_cpu.to(kDense);
  // create Tensor on cuda
  Tensor mat_sparse_cuda = mat_sparse_cpu.to(kCUDA);
  Tensor mat_dense_cuda = mat_dense_cpu.to(kCUDA);
  Tensor vec_dense_cuda = vec_dense_cpu.to(kCUDA);
  mat_sparse_cuda.set_coalesced(true);

  double time_ge = 0, time_sp = 0;
  Tensor result_spmv, result_gemv;

/////////CPU TEST//////////
{
  // std::cout << "CPU SPMV: ";
  timeval t1, t2;
  gettimeofday(&t1,NULL); 
  for (int i = 0; i < NUM_RUN; i++) {
    result_spmv = hice::matmul(mat_sparse_cpu, vec_dense_cpu);
  }
  gettimeofday(&t2,NULL); 
  time_sp = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / NUM_RUN;
}
{
  // std::cout << "CPU GEMV: ";
  timeval t1, t2;
  gettimeofday(&t1,NULL); 
  for (int i = 0; i < NUM_RUN; i++) {
    result_gemv = gemv_mklblas(mat_dense_cpu, vec_dense_cpu);
  }
  gettimeofday(&t2,NULL); 
  time_ge = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / NUM_RUN;
}
  ExpectEqualDenseWithError(result_spmv, result_gemv);
  double speedup_cpu = time_ge / time_sp;

/////////GPU TEST//////////
{
  // Warm-up calling before timing.
  Tensor result = matmul(mat_sparse_cuda, vec_dense_cuda);
  Tensor result1 = matmul(mat_dense_cuda, vec_dense_cuda);
}
{
  // std::cout << "CUDA SPMV: ";
  cuda_timer timer;
  timer.start();
  for (int i = 0; i < NUM_RUN; i++) {
    result_spmv = hice::matmul(mat_sparse_cuda, vec_dense_cuda);
  }
  time_sp = timer.stop() * 1.0 / NUM_RUN;
}
{
  // std::cout << "CUDA GEMV: ";
  cuda_timer timer;
  timer.start();
  for (int i = 0; i < NUM_RUN; i++) {
    result_gemv = gemv_cublas(mat_dense_cuda, vec_dense_cuda);
  }
  time_ge = timer.stop() * 1.0 / NUM_RUN;
}
  ExpectEqualDenseWithError(result_spmv, result_gemv);
  double speedup_gpu = time_ge / time_sp;

  std::cout << "\t[PASSED], CPU Speedup = " <<  speedup_cpu 
            << ", GPU Speedup = " << speedup_gpu << std::endl;
  std::cout << std::endl;
}