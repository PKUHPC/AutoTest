#pragma once
#include "hice/core/tensor.h"
#include "hice/ml/svm.h"
#include <cstring>

namespace hice {
  void get_working_set_cpu(const Tensor& data, const Tensor& data_row_idx, Tensor& data_rows,
                           int m, int n);
  void get_working_set_from_csr_cpu(const Tensor& d_data_csr, const Tensor& data_row_idx, Tensor& data_rows,
                      int m, int n);
  void get_working_set_from_csr_row_major_cpu(const Tensor& d_data_csr, const Tensor& data_row_idx, Tensor& data_rows,
                       int m, int n);
  void select_working_set_cpu(Tensor& ws_indicator, const Tensor& f_idx2sort, const Tensor& y,
                              const Tensor& alpha, double Cp, double Cn,
                              Tensor& working_set);
  void get_row_nnz_cpu(const Tensor& d_data_csr, Tensor& data_row_nnz, int data_rownum);
  void get_self_dot_cpu(const Tensor& tensor1, Tensor& self_dot);
  void get_self_dot_csr_cpu(const Tensor& tensor1, Tensor& self_dot, const Tensor& data_row_nnz);
  // void svm_matmul_cpu(Tensor tensor1, Tensor tensor2, Tensor result);
  // void svm_matmul_dense_csr_cpu(Tensor tensor2, Tensor tensor1, Tensor result);
  void svm_matmul_dense_csr_cpu(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);
  void RBF_kernel_cpu(const Tensor& self_dot0, const Tensor& self_dot1,
                      Tensor& dot_product, int m, int n, float gamma);
  void poly_kernel_cpu(Tensor& dot_product, float gamma, float coef0, int degree, int mn);
  void sigmoid_kernel_cpu(Tensor& dot_product, float gamma, float coef0, int mn);
  void sort_f_cpu(Tensor& f_val2sort, Tensor& f_idx2sort);
  void update_f_cpu(Tensor& f, const Tensor& alpha_diff, const Tensor &k_mat_rows,
                    int n_instances);
  void c_smo_solve_cpu(const Tensor& y, Tensor& f_val, Tensor& alpha,
                       Tensor& alpha_diff, const Tensor& working_set, double Cp,
                       double Cn, const Tensor& k_mat_rows, const Tensor& k_mat_diag,
                       int row_len, double eps, Tensor& diff, int max_iter);
  void svm_predict_cpu(Tensor& predict_result, const Tensor& predict_k_mat_rows, const Tensor& coef,
                       int predict_num_vects, int sv_size);

} // namespace hice
