#if 0

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
#include <numeric>
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/device/cuda/context_cuda.h"
#include "hice/ml/cuda/svm_kernel.cuh"

namespace hice {
__global__ void kernel_get_nnz_var(const int *data_row_nnz, float *var_data,
                                   float aver_nnz, int m) {
  KERNEL_LOOP(i, m) {
    var_data[i] = (data_row_nnz[i] - aver_nnz) * (data_row_nnz[i] - aver_nnz);
  }
}

__global__ void kernel_get_nnz_var_from_rowoffsets(const int *data_rowoffsets,
                                                   float *var_data,
                                                   float aver_nnz, int m) {
  KERNEL_LOOP(i, m) {
    var_data[i] = (data_rowoffsets[i + 1] - aver_nnz) *
                  (data_rowoffsets[i + 1] - aver_nnz);
  }
}
__global__ void kernel_get_working_set(const float *data, const int *data_row_idx,
                                       float *data_rows, int m, int n) {
  KERNEL_LOOP(i, m) {
    int row = data_row_idx[i];
    for (int j = row * n; j < (row + 1) * n; j++) {
      data_rows[i * n + j - row * n] = data[j];
    }
  }
}

__global__ void kernel_get_working_set_from_csr(
    const float *val, const int *col_ind, const int *row_ptr,
    const int *data_row_idx, float *data_rows, int m, int n) {
  KERNEL_LOOP(i, m) {
    int row = data_row_idx[i];
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      data_rows[col * m + i] = val[j];  // col major for cuSPARSE
    }
  }
}
__global__ void kernel_get_working_set_from_csr_row_major(
    const float *val, const int *col_ind, const int *row_ptr,
    const int *data_row_idx, float *data_rows, int m, int n) {
  KERNEL_LOOP(i, m) {
    int row = data_row_idx[i];
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      data_rows[i * n + col] = val[j];  // row major for cuSPARSE
    }
  }
}
__global__ void kernel_get_working_set_csr(
    const float *data, const int *data_rowoffsets,
    const int *data_colind,   // csr format of origin matrix
    const int *data_row_nnz,  // nnz per row
    const int *data_row_idx,  // row id
    float *data_rows,
    const int *data_rows_rowoffsets,
    int *data_rows_colind,  // csr format of sub matrix
    int m, int n, int data_rownum) {
  KERNEL_LOOP(i, data_rownum) {
    int row = data_row_idx[i];
    int num = data_row_nnz[row];
    int position_data = data_rowoffsets[row];
    int position_data_row = data_rows_rowoffsets[i];
    for (int j = 0; j < num; j++) {
      data_rows[position_data_row + j] = data[position_data + j];
      data_rows_colind[position_data_row + j] = data_colind[position_data + j];
    }
  }
}
__global__ void kernel_get_working_set_rowoffsets(const int *data_row_nnz,
                                                  const int *data_row_idx,
                                                  int *data_rows_rowoffsets,
                                                  int data_rownum) {
  KERNEL_LOOP(i, data_rownum) {
    int row = data_row_idx[i];
    data_rows_rowoffsets[i + 1] = data_row_nnz[row];
  }
}

__global__ void kernel_get_workingset_nnz_prow(const int *data_row_nnz,
                                               const int *data_row_idx,
                                               int *workingset_nnz_prow,
                                               int data_rownum) {
  KERNEL_LOOP(i, data_rownum) {
    int row = data_row_idx[i];
    workingset_nnz_prow[i] = data_row_nnz[row];
  }
}
__global__ void kernel_get_row_nnz(const int *data_rowoffsets, int *data_row_nnz,
                                   int data_rownum) {
  KERNEL_LOOP(i, data_rownum) {
    data_row_nnz[i] = data_rowoffsets[i + 1] - data_rowoffsets[i];
  }
}

__global__ void kernel_get_ws_indicator(int *ws_indicator,
                                        const int *working_set_first_half,
                                        int half_ws_size) {
  KERNEL_LOOP(i, half_ws_size) { ws_indicator[working_set_first_half[i]] = 1; }
}

__global__ void kernel_select_working_set(int *d_ws_indicator, const int *index,
                                          const int *y_data, const double *alpha_data,
                                          double Cp, double Cn,
                                          int *working_set_data,
                                          int n_instances, int ws_size) {
  int p_left = 0;
  int p_right = n_instances - 1;
  int n_selected = 0;
  while (n_selected < ws_size) {
    int i;
    if (p_left < n_instances) {
      i = index[p_left];
      while (d_ws_indicator[i] == 1 ||
             !is_I_up_device(alpha_data[i], y_data[i], Cp, Cn)) {
        // construct working set of I_up
        p_left++;
        if (p_left == n_instances) break;
        i = index[p_left];
      }
      if (p_left < n_instances) {
        working_set_data[n_selected++] = i;
        d_ws_indicator[i] = 1;
      }
    }
    if (p_right >= 0) {
      i = index[p_right];
      while (d_ws_indicator[i] == 1 ||
             !is_I_low_device(alpha_data[i], y_data[i], Cp, Cn)) {
        // construct working set of I_low
        p_right--;
        if (p_right == -1) break;
        i = index[p_right];
      }
      if (p_right >= 0) {
        working_set_data[n_selected++] = i;
        d_ws_indicator[i] = 1;
      }
    }
  }
}

__global__ void kernel_self_dot(const float *square_tensor1, float *self_dot, int n,
                                int m) {
  float sum = 0.0;
  KERNEL_LOOP(idx, n) {
    for (int i = idx * m; i < (idx + 1) * m; i++) sum += square_tensor1[i];
    self_dot[idx] = sum;
  }
}

__global__ void kernel_square(const float *tensor1, float *square_tensor1, int n) {
  KERNEL_LOOP(idx, n) { square_tensor1[idx] = tensor1[idx] * tensor1[idx]; }
}

__global__ void kernel_self_dot_csr(float *d_self_dot, float *values,
                                    int *column_indices, const int *row_offsets,
                                    const int *data_row_nnz, int num_rows,
                                    int num_cols) {
  KERNEL_LOOP(idx, num_rows) {
    int position_data = row_offsets[idx];
    int num_data = data_row_nnz[idx];
    for (int i = 0; i < num_data; i++)
      d_self_dot[idx] += values[position_data + i] * values[position_data + i];
  }
}
__global__ void kernel_RBF_kernel(const float *self_dot0, const float *self_dot1,
                                  float *dot_product, int m, int n,
                                  float gamma) {
  KERNEL_LOOP(idx, m * n) {
    int i = idx / n;  // i is row id
    int j = idx % n;  // j is column id
    dot_product[idx] =
        expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
  }
}

__global__ void kernel_RBF_kernel(const int *self_dot0_idx, const float *self_dot1,
                                  float *dot_product, int m, int n,
                                  float gamma) {
  KERNEL_LOOP(idx, m * n) {
    int i = idx / n;  // i is row id
    int j = idx % n;  // j is column id
    dot_product[idx] = expf(
        -(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) *
        gamma);
  }
}
__global__ void kernel_poly_kernel(float *dot_product, float gamma, float coef0,
                                   int degree, int mn) {
  KERNEL_LOOP(idx, mn) {
    dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
  }
}

__global__ void kernel_sigmoid_kernel(float *dot_product, float gamma,
                                      float coef0, int mn) {
  KERNEL_LOOP(idx, mn) {
    dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
  }
}
__global__ void update_f_kernel(double *f, int ws_size,
                                const double *alpha_diff,
                                const float *k_mat_rows, int n_instances) {
  KERNEL_LOOP(idx, n_instances) {  // one thread to update multiple fvalues.
    double sum_diff = 0;
    for (int i = 0; i < ws_size; ++i) {
      double d = alpha_diff[i];
      if (d != 0) {
        sum_diff += d * k_mat_rows[i * n_instances + idx];
      }
    }
    f[idx] -= sum_diff;
  }
}
void update_f(Tensor &f, const Tensor &alpha_diff, const Tensor &k_mat_rows,
              int n_instances) {
  KERNEL_LAUNCH(update_f_kernel, f.mutable_data<double>(), alpha_diff.size(),
                alpha_diff.data<double>(), k_mat_rows.data<float>(),
                n_instances);
}
template <typename T>
__device__ int get_block_min(const T *values, int *index) {
  int tid = threadIdx.x;
  index[tid] = tid;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (values[index[tid + offset]] < values[index[tid]]) {
        index[tid] = index[tid + offset];
      }
    }
    __syncthreads();
  }
  return index[0];
}
__global__ void c_smo_solve_kernel(const int *label, double *f_val,
                                   double *alpha, double *alpha_diff,
                                   const int *working_set, int ws_size,
                                   double Cp, double Cn,
                                   const float *k_mat_rows,
                                   const float *k_mat_diag, int row_len,
                                   double eps, double *diff, int max_iter) {
  extern __shared__ int shared_mem[];
  int *f_idx2reduce = shared_mem;
  double *f_val2reduce = (double *)&shared_mem[ws_size];
  double *alpha_i_diff =
      (double *)&shared_mem[ws_size + ws_size * sizeof(double) /
                                          sizeof(int)];  // delta alpha_i
  double *alpha_j_diff = &alpha_i_diff[1];
  float *kd = (float *)&alpha_j_diff[1];
  int tid = threadIdx.x;
  int wsi = working_set[tid];
  kd[tid] = k_mat_diag[wsi];
  double y = label[wsi];
  double f = f_val[wsi];
  double a = alpha[wsi];
  double aold = a;
  __syncthreads();
  double local_eps;
  int numOfIter = 0;
  while (1) {
    if (is_I_up_device(a, y, Cp, Cn))
      f_val2reduce[tid] = f;
    else
      f_val2reduce[tid] = INFINITY;
    int i = get_block_min(f_val2reduce, f_idx2reduce);
    double up_value = f_val2reduce[i];
    float kIwsI = k_mat_rows[row_len * i + wsi];
    __syncthreads();
    if (is_I_low_device(a, y, Cp, Cn))
      f_val2reduce[tid] = -f;
    else
      f_val2reduce[tid] = INFINITY;
    int j1 = get_block_min(f_val2reduce, f_idx2reduce);
    double low_value = -f_val2reduce[j1];
    double local_diff = low_value - up_value;
    if (numOfIter == 0) {
      local_eps = max(eps, 0.1f * local_diff);
      if (tid == 0) {
        diff[0] = local_diff;
      }
    }

    if (numOfIter > max_iter || local_diff < local_eps) {
      alpha[wsi] = a;
      alpha_diff[tid] = -(a - aold) * y;
      diff[1] = numOfIter;
      break;
    }
    __syncthreads();
    if (-up_value > -f && (is_I_low_device(a, y, Cp, Cn))) {
      double aIJ = kd[i] + kd[tid] - 2 * kIwsI;
      double bIJ = -up_value + f;
      f_val2reduce[tid] = (-bIJ * bIJ / aIJ);
    } else
      f_val2reduce[tid] = INFINITY;
    int j2 = get_block_min(f_val2reduce, f_idx2reduce);
    // update alpha
    if (tid == i) *alpha_i_diff = y > 0 ? Cp - a : a;
    if (tid == j2)
      *alpha_j_diff = min(y > 0 ? a : Cn - a,
                          (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
    __syncthreads();
    double l = min(*alpha_i_diff, *alpha_j_diff);
    if (tid == i) a += l * y;
    if (tid == j2) a -= l * y;
    float kJ2wsI = k_mat_rows[row_len * j2 + wsi];
    f -= l * (kJ2wsI - kIwsI);
    numOfIter++;
  }
}

__global__ void predict_kernel(int *predict_result,
                               const float *predict_k_mat_rows,
                               const double *coef, int predict_num_vects,
                               int sv_size) {
  KERNEL_LOOP(i, predict_num_vects) {
    double sum = 0.0;
    for (int j = 0; j < sv_size; j++) {
      sum += predict_k_mat_rows[i * sv_size + j] * coef[j];
    }
    predict_result[i] = sum > 0.0 ? 1 : -1;
  }
}

void svm_predict(Tensor &predict_result, const Tensor &predict_k_mat_rows,
                 const Tensor &coef, int predict_num_vects, int sv_size) {
  KERNEL_LAUNCH(predict_kernel, predict_result.mutable_data<int>(),
                predict_k_mat_rows.data<float>(), coef.data<double>(),
                predict_num_vects, sv_size);
}

void get_working_set(const Tensor &d_data, const Tensor &data_row_idx,
                     Tensor &data_rows, int m, int n) {
  KERNEL_LAUNCH(kernel_get_working_set, d_data.data<float>(),
                data_row_idx.data<int>(), data_rows.mutable_data<float>(), m,
                n);
}

void get_working_set_from_csr(const Tensor &d_data_csr,
                              const Tensor &data_row_idx, Tensor &data_rows,
                              int m, int n) {
  KERNEL_LAUNCH(kernel_get_working_set_from_csr,
                d_data_csr.values().data<float>(),
                d_data_csr.column_indices().data<int>(),
                d_data_csr.row_offsets().data<int>(), data_row_idx.data<int>(),
                data_rows.mutable_data<float>(), m, n);
}
void get_working_set_from_csr_row_major(const Tensor &d_data_csr,
                                        const Tensor &data_row_idx,
                                        Tensor &data_rows, int m, int n) {
  KERNEL_LAUNCH(kernel_get_working_set_from_csr_row_major,
                d_data_csr.values().data<float>(),
                d_data_csr.column_indices().data<int>(),
                d_data_csr.row_offsets().data<int>(), data_row_idx.data<int>(),
                data_rows.mutable_data<float>(), m, n);
}
// /** 从csr大稀疏矩阵中抽取data_row_idx标号对应的行，存为data_rows_csr**/
//
// void get_working_set_csr(Tensor &d_data_csr, Tensor &data_row_nnz,Tensor
// &data_row_idx, Tensor &data_rows_rowoffsets,Tensor &data_rows_colind,Tensor
// &data_rows_values,
//                      int m, int n, int data_rownum) {
//   KERNEL_LAUNCH(kernel_get_working_set_csr,
//   d_data_csr.data<float>(),d_data_csr.row_offsets(),d_data_csr.column_indices(),
//                      data_row_nnz.data<int>(), data_row_idx.data<int>(),
//                      data_rows_values.data<float>(),
//                      data_rows_rowoffsets.data<int>(),
//                      data_rows_colind.data<int>(), m, n, data_rownum);
// }
// void get_working_set_rowoffsets(Tensor &data_row_nnz, Tensor &data_row_idx,
//                                         Tensor &data_rows_rowoffsets,
//                                        int data_rownum)
// {
//     KERNEL_LAUNCH(kernel_get_working_set_rowoffsets,
//     data_row_nnz.data<int>(),
//                           data_row_idx.data<int>(),
//                           data_rows_rowoffsets.data<int>(),
//                           data_rownum);
//     thrust::inclusive_scan(thrust::cuda::par,
//     data_rows_rowoffsets.data<int>(), data_rows_rowoffsets.data<int>() +
//     data_rownum + 1, data_rows_rowoffsets.data<int>());
// }
//
// void get_working_set_rowoffsets1(Tensor &data_row_nnz, Tensor &data_row_idx,
//                                         Tensor &data_rows_rowoffsets,
//                                        int data_rownum)
// {
//     KERNEL_LAUNCH(kernel_get_working_set_rowoffsets,
//     data_row_nnz.data<int>(),
//                           data_row_idx.data<int>(),
//                           data_rows_rowoffsets.data<int>(),
//                           data_rownum);
//   //  thrust::inclusive_scan(thrust::cuda::par,
//   data_rows_rowoffsets.data<int>(), data_rows_rowoffsets.data<int>() +
//   data_rownum + 1, data_rows_rowoffsets.data<int>());
// }
//
// void get_working_set_rowoffsets2(Tensor &data_row_nnz, Tensor &data_row_idx,
//                                         Tensor &data_rows_rowoffsets,
//                                        int data_rownum)
// {
//     // KERNEL_LAUNCH(kernel_get_working_set_rowoffsets,
//     data_row_nnz.data<int>(),
//     //                       data_row_idx.data<int>(),
//     //                       data_rows_rowoffsets.data<int>(),
//     //                       data_rownum);
//   thrust::inclusive_scan(thrust::cuda::par, data_rows_rowoffsets.data<int>(),
//   data_rows_rowoffsets.data<int>() + data_rownum + 1,
//   data_rows_rowoffsets.data<int>());
// }

void get_workingset_nnz_prow(const Tensor &data_row_nnz,
                             const Tensor &data_row_idx,
                             Tensor &workingset_nnz_prow, int data_rownum) {
  KERNEL_LAUNCH(kernel_get_workingset_nnz_prow, data_row_nnz.data<int>(),
                data_row_idx.data<int>(),
                workingset_nnz_prow.mutable_data<int>(), data_rownum);
}
void get_row_nnz(const Tensor &d_data_csr, Tensor &data_row_nnz,
                 int data_rownum) {
  KERNEL_LAUNCH(kernel_get_row_nnz, d_data_csr.row_offsets().data<int>(),
                data_row_nnz.mutable_data<int>(), data_rownum);
}

void RBF_kernel(const Tensor &self_dot0, const Tensor &self_dot1,
                Tensor &dot_product, int m, int n, float gamma) {
  ScalarType sc_type = self_dot0.scalar_type();
  auto *d_self_dot1 = self_dot1.data<float>();
  auto *d_dot_product = dot_product.mutable_data<float>();
  if (sc_type == ScalarType::Float) {
    using scalar_t = float;
    auto *d_self_dot0 = self_dot0.data<scalar_t>();
    KERNEL_LAUNCH(kernel_RBF_kernel, d_self_dot0, d_self_dot1, d_dot_product, m,
                  n, gamma);
  } else if (sc_type == ScalarType::Int32) {
    using scalar_t = int;
    auto *d_self_dot0 = self_dot0.data<scalar_t>();
    KERNEL_LAUNCH(kernel_RBF_kernel, d_self_dot0, d_self_dot1, d_dot_product, m,
                  n, gamma);
  }
}

void poly_kernel(Tensor &dot_product, float gamma, float coef0, int degree,
                 int mn) {
  KERNEL_LAUNCH(kernel_poly_kernel, dot_product.mutable_data<float>(), gamma,
                coef0, degree, mn);
}

void sigmoid_kernel(Tensor &dot_product, float gamma, float coef0, int mn) {
  KERNEL_LAUNCH(kernel_sigmoid_kernel, dot_product.mutable_data<float>(), gamma,
                coef0, mn);
}
void svm_matmul_csr_dense_cuda(const Tensor &tensor1, const Tensor &tensor2,
                               Tensor &result) {
  // tensor1 csr / tensor2 dense
  float alpha = 1.0;
  float beta = 0.0;
  int num_rows = tensor1.dim(0);
  int num_cols = tensor1.dim(1);
  int nnz = tensor1.size();
  int *row_offsets = const_cast<Tensor &>(tensor1).mutable_row_offsets().mutable_data<int>();
  int *column_indices = const_cast<Tensor &>(tensor1).mutable_column_indices().mutable_data<int>();
  float *values = const_cast<Tensor &>(tensor1).mutable_values().mutable_data<float>();
  float *tensor2_ptr = const_cast<Tensor &>(tensor2).mutable_data<float>();
  float *result_ptr = result.mutable_data<float>();

  int m = tensor1.dim(0);
  int n = tensor2.dim(0);
  int k = tensor1.dim(1);

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descrA = 0;
  // cusparseMatDescr_t descrB = 0;
  // cusparseMatDescr_t descrC = 0;
  // initialize cusparse library
  cusparseCreate(&handle);
  // create and setup matrix descriptors A, B & C
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descrA,
                  values, row_offsets, column_indices, tensor2_ptr, n, &beta,
                  result_ptr, m);
}

void get_self_dot(const Tensor &train_data, Tensor &self_dot) {
  auto dim0_tensor1 = train_data.dim(0);
  auto dim1_tensor1 = train_data.dim(1);
  Tensor square_tensor1 =
      full({dim0_tensor1, dim1_tensor1}, 0, device(kCUDA).dtype(kFloat));
  cudaDeviceSynchronize();
  KERNEL_LAUNCH(kernel_square, train_data.data<float>(),
                square_tensor1.mutable_data<float>(),
                dim0_tensor1 * dim1_tensor1);
  cudaDeviceSynchronize();
  KERNEL_LAUNCH(kernel_self_dot, square_tensor1.data<float>(),
                self_dot.mutable_data<float>(), dim0_tensor1, dim1_tensor1);
}

void get_self_dot_csr(const Tensor &tensor1, Tensor &self_dot,
                      const Tensor &data_row_nnz) {
  int num_rows = tensor1.dim(0);
  int num_cols = tensor1.dim(1);
  int nnz = tensor1.size();
  int *row_offsets = const_cast<Tensor &>(tensor1).mutable_row_offsets().mutable_data<int>();
  int *column_indices = const_cast<Tensor &>(tensor1).mutable_column_indices().mutable_data<int>();
  float *values = const_cast<Tensor &>(tensor1).mutable_values().mutable_data<float>();
  float *d_self_dot = self_dot.mutable_data<float>();
  KERNEL_LAUNCH(kernel_self_dot_csr, d_self_dot, values, column_indices,
                row_offsets, data_row_nnz.data<int>(), num_rows, num_cols);
}

void c_smo_solve(const Tensor &y, Tensor &f_val, Tensor &alpha,
                 Tensor &alpha_diff, const Tensor &working_set, double Cp,
                 double Cn, const Tensor &k_mat_rows, const Tensor &k_mat_diag,
                 int row_len, double eps, Tensor &diff, int max_iter) {
  size_t ws_size = working_set.size();
  size_t smem_size = 0;
  smem_size += ws_size * sizeof(int);     // f_idx2reduce
  smem_size += ws_size * sizeof(double);  // f_val2reduce
  smem_size += ws_size * sizeof(float);   // kd
  smem_size += 2 * sizeof(double);        // alpha diff
  c_smo_solve_kernel<<<1, ws_size, smem_size>>>(
      y.data<int>(), f_val.mutable_data<double>(), alpha.mutable_data<double>(),
      alpha_diff.mutable_data<double>(), working_set.data<int>(), ws_size, Cp,
      Cn, k_mat_rows.data<float>(), k_mat_diag.data<float>(), row_len, eps,
      diff.mutable_data<double>(), max_iter);
}
void sort_f(Tensor &d_f_val2sort, Tensor &d_f_idx2sort) {
  thrust::sort_by_key(thrust::cuda::par, d_f_val2sort.mutable_data<double>(),
                      d_f_val2sort.mutable_data<double>() + d_f_val2sort.size(),
                      d_f_idx2sort.mutable_data<int>(), thrust::less<double>());
}
void select_working_set(Tensor &ws_indicator, const Tensor &f_idx2sort,
                        const Tensor &y, const Tensor &alpha, double Cp,
                        double Cn, Tensor &working_set) {
  int n_instances = ws_indicator.size();
  int ws_size = working_set.size();
  kernel_select_working_set<<<1, 1>>>(
      ws_indicator.mutable_data<int>(), f_idx2sort.data<int>(), y.data<int>(),
      alpha.data<double>(), Cp, Cn, working_set.mutable_data<int>(),
      n_instances, ws_size);
}
void get_ws_indicator(Tensor &ws_indicator,
                      const Tensor &working_set_first_half, int half_ws_size) {
  kernel_get_ws_indicator<<<1, half_ws_size>>>(
      ws_indicator.mutable_data<int>(), working_set_first_half.data<int>(),
      half_ws_size);
}
}  // namespace hice

#endif