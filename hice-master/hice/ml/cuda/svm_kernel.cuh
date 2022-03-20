#pragma once
#include "hice/core/tensor.h"
#include "hice/ml/svm.h"
namespace hice {

const int SVM_BLOCK_SIZE = 512;

const int SVM_NUM_BLOCKS = 32 * 56;

#define KERNEL_LAUNCH(kernel_name, ...)                                   \
  kernel_name<<<SVM_NUM_BLOCKS, SVM_BLOCK_SIZE>>>(__VA_ARGS__);

#define KERNEL_LOOP(i, n)                                                      \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
        i += blockDim.x * gridDim.x)

__device__ inline bool is_I_up_device(double a, double y, double Cp,
                                        double Cn) {
  return (y > 0 && a < Cp) || (y < 0 && a > 0);
}

__device__ inline bool is_I_low_device(double a, double y, double Cp,
                                         double Cn) {
  return (y > 0 && a > 0) || (y < 0 && a < Cn);
}

__device__ inline bool is_free_device(double a, double y, double Cp,
                                        double Cn) {
  return a > 0 && (y > 0 ? a < Cp : a < Cn);
}

template<typename T>
__host__ __device__ T getgriddim(T totallen, T blockdim)
{
    return (totallen + blockdim - (T)1) / blockdim;
}

void RBF_kernel(const Tensor& self_dot0, const Tensor& self_dot1, Tensor &dot_product,
                int m, int n, float gamma);

void sigmoid_kernel(Tensor& dot_product, float gamma, float coef0, int mn);
void poly_kernel(Tensor& dot_product, float gamma, float coef0, int degree, int mn);

// void svm_matmul(Tensor tensor1, Tensor tensor2, Tensor result);
// void svm_matmul_trans(Tensor tensor1, Tensor tensor2, Tensor result);
void svm_matmul_csr_dense_cuda(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
void get_self_dot(const Tensor& train_data, Tensor& self_dot);
void get_self_dot_csr(const Tensor& tensor1, Tensor& self_dot, const Tensor& data_row_nnz);
void get_working_set(const Tensor& d_data, const Tensor& data_row_idx, Tensor& data_rows,
                     int m, int n);
void get_working_set_from_csr(const Tensor& d_data_csr, const Tensor& data_row_idx,
                              Tensor& data_rows, int m, int n);
void get_working_set_from_csr_row_major(const Tensor& d_data_csr, const Tensor& data_row_idx, Tensor& data_rows,
                      int m, int n);
void get_row_nnz(const Tensor& d_data_csr, Tensor& data_row_nnz, int data_rownum);
void get_workingset_nnz_prow(const Tensor& data_row_nnz, const Tensor& data_row_idx,
                            Tensor& workingset_nnz_prow, int data_rownum);
void c_smo_solve(const Tensor& y, Tensor& f_val, Tensor& alpha, Tensor& alpha_diff,
                  const Tensor& working_set, double Cp, double Cn, const Tensor& k_mat_rows,
                  const Tensor& k_mat_diag, int row_len, double eps, Tensor& diff,
                  int max_iter);
void sort_f(Tensor& d_f_val2sort, Tensor& d_f_idx2sort);
void select_working_set(Tensor& ws_indicator, const Tensor& f_idx2sort, const Tensor& y,
                        const Tensor& alpha, double Cp, double Cn,
                        Tensor& working_set);
void update_f(Tensor& f, const Tensor& alpha_diff, const Tensor& k_mat_rows,
              int n_instances);
void get_ws_indicator(Tensor& ws_indicator, const Tensor& working_set_first_half,
                      int half_ws_size);
void svm_predict(Tensor& predict_result, const Tensor& predict_k_mat_rows, const Tensor& coef,
                      int predict_num_vects, int sv_size);
} // namespace hice
