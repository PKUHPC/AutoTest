#include "hice/ml/cpu/svm_kernel.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"

#ifdef HICE_USE_MKL
#include "hice/device/cpu/common_mkl.h"
#endif

namespace hice {

void c_smo_solve_kernel_cpu(const int *label, double *f_val, double *alpha,
                            double *alpha_diff, const int *working_set,
                            int ws_size, double Cp, double Cn,
                            const float *k_mat_rows, const float *k_mat_diag,
                            int row_len, double eps, double *diff,
                            int max_iter) {
  double alpha_i_diff;
  double alpha_j_diff;
  std::vector<float> kd(ws_size);
  std::vector<double> a_old(ws_size);
  std::vector<double> kIwsI(ws_size);
  std::vector<double> f(ws_size);
  std::vector<double> y(ws_size);
  std::vector<double> a(ws_size);
  for (int tid = 0; tid < ws_size; ++tid) {
    int wsi = working_set[tid];
    f[tid] = f_val[wsi];
    a_old[tid] = a[tid] = alpha[wsi];
    y[tid] = label[wsi];
    kd[tid] = k_mat_diag[wsi];
  }
  double local_eps;
  int numOfIter = 0;
  while (1) {
    int i = 0;
    double up_value = INFINITY;
    for (int tid = 0; tid < ws_size; ++tid) {
      if (is_I_up(a[tid], y[tid], Cp, Cn))
        if (f[tid] < up_value) {
          up_value = f[tid];
          i = tid;
        }
    }
    for (int tid = 0; tid < ws_size; ++tid) {
      kIwsI[tid] = k_mat_rows[row_len * i + working_set[tid]];
    }
    double low_value = -INFINITY;
    for (int tid = 0; tid < ws_size; ++tid) {
      if (is_I_low(a[tid], y[tid], Cp, Cn))
        if (f[tid] > low_value) {
          low_value = f[tid];
        }
    }
    double local_diff = low_value - up_value;
    if (numOfIter == 0) {
      local_eps = std::max(eps, 0.1f * local_diff);
      diff[0] = local_diff;
    }
    if (numOfIter > max_iter || local_diff < local_eps) {
      for (int tid = 0; tid < ws_size; ++tid) {
        int wsi = working_set[tid];
        alpha_diff[tid] = -(a[tid] - a_old[tid]) * y[tid];
        alpha[wsi] = a[tid];
      }
      diff[1] = numOfIter;
      break;
    }
    int j2 = 0;
    double min_t = INFINITY;
    for (int tid = 0; tid < ws_size; ++tid) {
      if (-up_value > -f[tid] && (is_I_low(a[tid], y[tid], Cp, Cn))) {
        double aIJ = kd[i] + kd[tid] - 2 * kIwsI[tid];
        double bIJ = -up_value + f[tid];
        double ft = -bIJ * bIJ / aIJ;
        if (ft < min_t) {
          min_t = ft;
          j2 = tid;
        }
      }
    }
    alpha_i_diff = y[i] > 0 ? Cp - a[i] : a[i];
    alpha_j_diff =
        std::min(y[j2] > 0 ? a[j2] : Cn - a[j2],
                 (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
    double l = std::min(alpha_i_diff, alpha_j_diff);
    a[i] += l * y[i];
    a[j2] -= l * y[j2];
    for (int tid = 0; tid < ws_size; ++tid) {
      int wsi = working_set[tid];
      double kJ2wsI = k_mat_rows[row_len * j2 + wsi];  // K[J2, wsi]
      f[tid] -= l * (kJ2wsI - kIwsI[tid]);
    }
    numOfIter++;
  }
}

void get_row_nnz_cpu(const Tensor &d_data_csr, Tensor &data_row_nnz,
                     int data_rownum) {
  const int *data_rowoffsets = d_data_csr.row_offsets().data<int>();
  int *data_row_nnz_ptr = data_row_nnz.mutable_data<int>();
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < data_rownum; i++) {
    data_row_nnz_ptr[i] = data_rowoffsets[i + 1] - data_rowoffsets[i];
  }
}

void get_working_set_cpu(const Tensor &data, const Tensor &data_row_idx,
                         Tensor &data_rows, int m, int n) {
  const float *d = data.data<float>();
  const int *d_r_idx = data_row_idx.data<int>();
  float *d_r = data_rows.mutable_data<float>();
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < m; i++) {
    int row = d_r_idx[i];
    for (int j = row * n; j < (row + 1) * n; j++) {
      d_r[i * n + j - row * n] = d[j];
    }
  }
}
void get_working_set_from_csr_cpu(const Tensor &d_data_csr,
                                  const Tensor &data_row_idx, Tensor &data_rows,
                                  int m, int n) {
  const float *val = d_data_csr.values().data<float>();
  const int *col_ind = d_data_csr.column_indices().data<int>();
  const int *row_ptr = d_data_csr.row_offsets().data<int>();
  const int *data_row_idx_ptr = data_row_idx.data<int>();
  float *data_rows_ptr = data_rows.mutable_data<float>();
  for (int i = 0; i < m; i++) {
    int row = data_row_idx_ptr[i];
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      // data_rows[i * n + col] = val[j]; // col-major for cuSPARSE
      data_rows_ptr[col * m + i] = val[j];  // col-major for cuSPARSE
    }
  }
}
void get_working_set_from_csr_row_major_cpu(const Tensor &d_data_csr,
                                            const Tensor &data_row_idx,
                                            Tensor &data_rows, int m, int n) {
  const float *val = d_data_csr.values().data<float>();
  const int *col_ind = d_data_csr.column_indices().data<int>();
  const int *row_ptr = d_data_csr.row_offsets().data<int>();
  const int *data_row_idx_ptr = data_row_idx.data<int>();
  float *data_rows_ptr = data_rows.mutable_data<float>();
  for (int i = 0; i < m; i++) {
    int row = data_row_idx_ptr[i];
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      data_rows_ptr[i * n + col] = val[j];  // col-major for cuSPARSE
      // data_rows_ptr[col * m + i] = val[j]; // col-major for cuSPARSE
    }
  }
}
void get_self_dot_csr_cpu(const Tensor &tensor1, Tensor &self_dot,
                          const Tensor &data_row_nnz) {
  int num_rows = tensor1.dim(0);
  int num_cols = tensor1.dim(1);
  int nnz = tensor1.size();
  const int *row_offsets = tensor1.row_offsets().data<int>();
  const int *column_indices = tensor1.column_indices().data<int>();
  const float *values = tensor1.values().data<float>();
  float *d_self_dot = self_dot.mutable_data<float>();
  const int *data_row_nnz_ptr = data_row_nnz.data<int>();
#pragma omp parallel for schedule(guided)
  for (int idx = 0; idx < num_rows; idx++) {
    int position_data = row_offsets[idx];
    int num_data = data_row_nnz_ptr[idx];
    for (int i = 0; i < num_data; i++)
      d_self_dot[idx] += values[position_data + i] * values[position_data + i];
  }
}
void get_self_dot_cpu(const Tensor &tensor1, Tensor &self_dot) {
  long int num_rows = tensor1.dim(0);
  long int num_cols = tensor1.dim(1);
  auto size = tensor1.size();
  Tensor square_tensor1 =
      full({num_rows, num_cols}, 0, device(kCPU).dtype(kFloat));
  const float *t1 = tensor1.data<float>();
  float *s_t = square_tensor1.mutable_data<float>();
  float *s_d = self_dot.mutable_data<float>();
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < size; i++) {
    s_t[i] = t1[i] * t1[i];
  }
#pragma omp parallel for schedule(guided)
  for (int idx = 0; idx < num_rows; idx++) {
    float sum = 0.0;
#pragma omp parallel for schedule(guided)
    for (int i = idx * num_cols; i < (idx + 1) * num_cols; i++) sum += s_t[i];
    s_d[idx] = sum;
  }
}
void select_working_set_cpu(Tensor &ws_indicator, const Tensor &f_idx2sort,
                            const Tensor &y, const Tensor &alpha, double Cp,
                            double Cn, Tensor &working_set) {
  int n_instances = ws_indicator.size();
  int p_left = 0;
  int p_right = n_instances - 1;
  int n_selected = 0;
  const int *index = f_idx2sort.data<int>();
  const int *y_data = y.data<int>();
  const double *alpha_data = alpha.data<double>();
  int *working_set_data = working_set.mutable_data<int>();
  int *ws_ind_data = ws_indicator.mutable_data<int>();
  while (n_selected < working_set.size()) {
    int i;
    if (p_left < n_instances) {
      i = index[p_left];
      while (ws_ind_data[i] == 1 ||
             !is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
        p_left++;
        if (p_left == n_instances) break;
        i = index[p_left];
      }
      if (p_left < n_instances) {
        working_set_data[n_selected++] = i;
        ws_ind_data[i] = 1;
      }
    }
    if (p_right >= 0) {
      i = index[p_right];
      while (ws_ind_data[i] == 1 ||
             !is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
        p_right--;
        if (p_right == -1) break;
        i = index[p_right];
      }
      if (p_right >= 0) {
        working_set_data[n_selected++] = i;
        ws_ind_data[i] = 1;
      }
    }
  }
}

// void svm_matmul_cpu(Tensor tensor1, Tensor tensor2, Tensor result) {
//   CBLAS_ORDER Order = CblasRowMajor;
//   CBLAS_TRANSPOSE TransA = CblasNoTrans;
//   CBLAS_TRANSPOSE TransB = CblasNoTrans;
//
//   float alpha = 1.0;
//   float beta = 0.0;
//   float *d_a = tensor1.data<float>();
//   float *d_b = tensor2.data<float>();
//   float *d_r = result.data<float>();
//
//   int m = tensor1.dim(0);
//   int n = tensor2.dim(1);
//   int k = tensor1.dim(1);
//   cblas_sgemm(Order, TransA, TransB, m, n, k, alpha, d_a, k, d_b, n, beta,
//   d_r, n);
//
// }

void svm_matmul_dense_csr_cpu(const Tensor &tensor1, const Tensor &tensor2,
                              Tensor &result) {
  // tensor1: csr
  // tensor2 : dense
  int num_rows = tensor1.dim(0);
  int num_cols = tensor1.dim(1);
  int *row_offsets =
      const_cast<Tensor &>(tensor1).mutable_row_offsets().mutable_data<int>();
  int *column_indices =
      const_cast<Tensor &>(tensor1).mutable_column_indices().mutable_data<int>();
  float *values =
      const_cast<Tensor &>(tensor1).mutable_values().mutable_data<float>();
  const float *tensor2_ptr = tensor2.data<float>();
  float *result_ptr = result.mutable_data<float>();
#ifdef HICE_USE_MKL
  float alpha = 1.0;
  float beta = 0.0;
  sparse_matrix_t csrA;
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, num_rows, num_cols,
                          row_offsets, row_offsets + 1, column_indices, values);
  //
  mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA,
                  SPARSE_LAYOUT_COLUMN_MAJOR, tensor2_ptr, tensor2.dim(0),
                  tensor2.dim(1), beta, result_ptr, num_rows);
  // mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA,
  //                 SPARSE_LAYOUT_ROW_MAJOR, tensor2_ptr, tensor1.dim(0),
  //                 tensor2.dim(1), beta, result_ptr, num_rows);
  mkl_sparse_destroy(csrA);
#else
  HICE_LOG(ERROR) << "HICE does not support mkl";
#endif
}

void RBF_kernel_cpu(const Tensor &self_dot0, const Tensor &self_dot1,
                    Tensor &dot_product, int m, int n, float gamma) {
  ScalarType sc_type = self_dot0.scalar_type();
  float *dot_product_data = dot_product.mutable_data<float>();
  const float *self_dot1_data = self_dot1.data<float>();
  if (sc_type == ScalarType::Int32) {
    const int *self_dot0_idx_data = self_dot0.data<int>();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; ++j) {
        dot_product_data[i * n + j] =
            expf(-(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] -
                   dot_product_data[i * n + j] * 2) *
                 gamma);
      }
    }
  } else if (sc_type == ScalarType::Float) {
    const float *self_dot0_data = self_dot0.data<float>();
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; ++j) {
        dot_product_data[i * n + j] =
            expf(-(self_dot0_data[i] + self_dot1_data[j] -
                   dot_product_data[i * n + j] * 2) *
                 gamma);
      }
    }
  }
}
void poly_kernel_cpu(Tensor &dot_product, float gamma, float coef0, int degree,
                     int mn) {
  float *dot_product_data = dot_product.mutable_data<float>();
#pragma omp parallel for schedule(guided)
  for (int idx = 0; idx < mn; idx++) {
    dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
  }
}

void sigmoid_kernel_cpu(Tensor &dot_product, float gamma, float coef0, int mn) {
  float *dot_product_data = dot_product.mutable_data<float>();
#pragma omp parallel for schedule(guided)
  for (int idx = 0; idx < mn; idx++) {
    dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
  }
}

void c_smo_solve_cpu(const Tensor &y, Tensor &f_val, Tensor &alpha,
                     Tensor &alpha_diff, const Tensor &working_set, double Cp,
                     double Cn, const Tensor &k_mat_rows,
                     const Tensor &k_mat_diag, int row_len, double eps,
                     Tensor &diff, int max_iter) {
  c_smo_solve_kernel_cpu(y.data<int>(), f_val.mutable_data<double>(),
                         alpha.mutable_data<double>(),
                         alpha_diff.mutable_data<double>(),
                         working_set.data<int>(), working_set.size(), Cp, Cn,
                         k_mat_rows.data<float>(), k_mat_diag.data<float>(),
                         row_len, eps, diff.mutable_data<double>(), max_iter);
}
void sort_f_cpu(Tensor &f_val2sort, Tensor &f_idx2sort) {
  std::vector<std::pair<double, int>> paris;
  double *f_val2sort_data = f_val2sort.mutable_data<double>();
  int *f_idx2sort_data = f_idx2sort.mutable_data<int>();
  for (int i = 0; i < f_val2sort.size(); ++i) {
    paris.emplace_back(f_val2sort_data[i], f_idx2sort_data[i]);
  }
  std::sort(paris.begin(), paris.end());
  for (int i = 0; i < f_idx2sort.size(); ++i) {
    f_idx2sort_data[i] = paris[i].second;
  }
}

void update_f_cpu(Tensor &f, const Tensor &alpha_diff, const Tensor &k_mat_rows,
                  int n_instances) {
  double *f_data = f.mutable_data<double>();
  const double *alpha_diff_data = alpha_diff.data<double>();
  const float *k_mat_rows_data = k_mat_rows.data<float>();
#pragma omp parallel for schedule(guided)
  for (int idx = 0; idx < n_instances; ++idx) {
    double sum_diff = 0;
    for (int i = 0; i < alpha_diff.size(); ++i) {
      double d = alpha_diff_data[i];
      if (d != 0) {
        sum_diff += d * k_mat_rows_data[i * n_instances + idx];
      }
    }
    f_data[idx] -= sum_diff;
  }
}
void svm_predict_cpu(Tensor &predict_result, const Tensor &predict_k_mat_rows,
                     const Tensor &coef, int predict_num_vects, int sv_size) {
  int *predict_result_data = predict_result.mutable_data<int>();
  const float *predict_k_mat_rows_data = predict_k_mat_rows.data<float>();
  const double *coef_data = coef.data<double>();
  for (int i = 0; i < predict_num_vects; i++) {
    double sum = 0.0;
    for (int j = 0; j < sv_size; j++) {
      sum += predict_k_mat_rows_data[i * sv_size + j] * coef_data[j];
    }
    predict_result_data[i] = sum > 0.0 ? 1 : -1;
  }
}
}  // namespace hice
