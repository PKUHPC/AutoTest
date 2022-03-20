#if 0
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/ml/svm.h"
#include "hice/ml/cuda/svm_kernel.cuh"
#include "hice/basic/transpose.h"
#include "hice/basic/reshape.h"
#include "hice/basic/copy.h"
#include "hice/math/matmul.h"

namespace hice {


namespace{

void svm_cuda_csr(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor &result,
         SvmParam param){
  //  std::cout << "=========Kernel: svm_cuda_csr==========" << std::endl;

   TensorPrinter tp;
   auto num_vects = train_data.dim(0);
   auto dim_vects = train_data.dim(1);
   double rho = 0.0;
   auto ws_size = get_working_set_size(num_vects, dim_vects);
   auto half_ws_size = ws_size / 2;
   int same_local_diff_cnt = 0;
   double previous_local_diff = INFINITY;
   int swap_local_diff_cnt = 0;
   double last_local_diff = INFINITY;
   double second_last_local_diff = INFINITY;
   double Cp = param.C;
   double Cn = param.C;
   double gamma = param.gamma;
   double eps = param.epsilon;
   int degree = param.degree;
   double coef0 = param.coef0;
   Tensor h_label = label.to(kCPU);
   Tensor alpha=full({num_vects}, 0, device(kCUDA).dtype(kDouble));


   Tensor d_self_dot = full({num_vects, 1}, 0, device(kCUDA).dtype(kFloat));
   Tensor d_self_dot_csr = full({num_vects}, 0, device(kCUDA).dtype(kFloat));
   Tensor data_row_nnz = full({num_vects}, 0, device(kCUDA).dtype(kInt32));
   get_row_nnz(train_data, data_row_nnz, num_vects);
   // tp.print(data_row_nnz);
   get_self_dot_csr(train_data, d_self_dot_csr, data_row_nnz);
   // tp.print(d_self_dot_csr);
   Tensor d_k_mat_rows = full({ws_size, num_vects}, 0, device(kCUDA).dtype(kFloat));
   Tensor d_k_mat_rows_first_half = wrap({half_ws_size, num_vects}, d_k_mat_rows.mutable_data<float>(),
            device(kCUDA).dtype(kFloat), false);
   Tensor d_k_mat_rows_last_half = wrap({half_ws_size, num_vects},
            d_k_mat_rows.mutable_data<float>() + d_k_mat_rows.size() / 2,
            device(kCUDA).dtype(kFloat), false);
   Tensor d_working_set = full({1, ws_size}, 0, device(kCUDA).dtype(kInt32));
   Tensor d_working_set_first_half = wrap({1, half_ws_size}, d_working_set.mutable_data<int>(),
                                         device(kCUDA).dtype(kInt32), false);
   Tensor d_working_set_last_half = wrap({1, half_ws_size},
                                        d_working_set.mutable_data<int>() + d_working_set.size() / 2,
                                       device(kCUDA).dtype(kInt32), false);
   Tensor f_val({1, num_vects}, device(kCPU).dtype(kDouble));
   Tensor f_idx({1, num_vects}, device(kCPU).dtype(kInt32));
   Tensor diff = full({1, 2}, 0, device(kCPU).dtype(kDouble));
   for (int i = 0; i < num_vects; ++i) {
     f_idx.mutable_data<int>()[i] = i;
     f_val.mutable_data<double>()[i] = -1 * h_label.data<int>()[i];
   }
   Tensor d_f_val = f_val.to(kCUDA);
   Tensor d_f_idx = f_idx.to(kCUDA);
   Tensor d_diff = diff.to(kCUDA);
   Tensor d_f_idx2sort = full({1, num_vects}, 0, device(kCUDA).dtype(kInt32));
   Tensor d_f_val2sort = full({1, num_vects}, 0, device(kCUDA).dtype(kDouble));
   Tensor d_alpha_diff = full({1, ws_size}, 0, device(kCUDA).dtype(kDouble));
   Tensor d_kmat_diag = full({1, num_vects}, 1, device(kCUDA).dtype(kFloat));
   Tensor d_data_rows =
       full({ws_size, dim_vects}, 0, device(kCUDA).dtype(kFloat));
   Tensor d_data_rows_half =
       full({half_ws_size, dim_vects}, 0, device(kCUDA).dtype(kFloat));
   Tensor ws_indicator({num_vects, 1}, device(kCUDA).dtype(kInt32));
   int *ws_ind = ws_indicator.mutable_data<int>();
   int ws_ind_size = sizeof(int) * num_vects;
   long long local_iter = 0;
   int out_max_iter = -1;
   int max_iter = std::max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
  //  std::cout << "traing start" << std::endl;
#if 1
   for (int iter = 0;; ++iter)  // global_iter
   {
    copy(d_f_idx, d_f_idx2sort);
    copy(d_f_val, d_f_val2sort);
    sort_f(d_f_val2sort, d_f_idx2sort);
    cudaMemset(ws_ind, 0, ws_ind_size);
    if (0 == iter) {
      select_working_set(ws_indicator, d_f_idx2sort, label, alpha, Cp, Cn,
                         d_working_set);
      get_working_set_from_csr(train_data, d_working_set, d_data_rows, ws_size, dim_vects);
      cudaDeviceSynchronize();
      svm_matmul_csr_dense_cuda(train_data, d_data_rows, d_k_mat_rows);
      cudaDeviceSynchronize();
      // tp.print(d_k_mat_rows);
      if(param.kernel_type == SvmParam::RBF){
        RBF_kernel(d_working_set, d_self_dot_csr, d_k_mat_rows,
                  ws_size, num_vects, gamma);
      }else if(param.kernel_type == SvmParam::POLY){
        poly_kernel(d_k_mat_rows, gamma, coef0, degree, ws_size*num_vects);
      }

    } else {
      copy(d_k_mat_rows_last_half, d_k_mat_rows_first_half);
      copy(d_working_set_last_half, d_working_set_first_half);
      get_ws_indicator(ws_indicator, d_working_set_first_half, half_ws_size);
      select_working_set(ws_indicator, d_f_idx2sort, label, alpha, Cp, Cn,
                         d_working_set_last_half);
      get_working_set_from_csr(train_data, d_working_set_last_half, d_data_rows_half,
                      half_ws_size, dim_vects);
      cudaDeviceSynchronize();
      // svm_matmul(d_data_rows_half, train_data_T, d_k_mat_rows_last_half);
      // matmul(d_data_rows_half, train_data_T, d_k_mat_rows_last_half, kNoTrans, kNoTrans);
      svm_matmul_csr_dense_cuda(train_data, d_data_rows_half, d_k_mat_rows_last_half);
      cudaDeviceSynchronize();
      // tp.print(d_k_mat_rows);
      if(param.kernel_type == SvmParam::RBF){
        RBF_kernel(d_working_set_last_half, d_self_dot_csr, d_k_mat_rows_last_half,
                   half_ws_size, num_vects, gamma);
      }else if(param.kernel_type == SvmParam::POLY){
        poly_kernel(d_k_mat_rows_last_half, gamma, coef0, degree, half_ws_size*num_vects);
      }
    }
    c_smo_solve(label, d_f_val, alpha, d_alpha_diff, d_working_set, Cp, Cn,
                d_k_mat_rows, d_kmat_diag, num_vects, eps, d_diff, max_iter);
    update_f(d_f_val, d_alpha_diff, d_k_mat_rows, num_vects);
    cudaDeviceSynchronize();
    f_val = d_f_val.to(kCPU);
    diff = d_diff.to(kCPU);
    const double *diff_data = diff.data<double>();
    local_iter += diff_data[1];
    if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
      same_local_diff_cnt++;
    } else {
      same_local_diff_cnt = 0;
      previous_local_diff = diff_data[0];
    }
    if (fabs(diff_data[0] - second_last_local_diff) < eps * 0.001) {
      swap_local_diff_cnt++;
    } else {
      swap_local_diff_cnt = 0;
    }
    second_last_local_diff = last_local_diff;
    last_local_diff = diff_data[0];
    if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) ||
        diff_data[0] < eps || (out_max_iter != -1) && (iter == out_max_iter) ||
        (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
      Tensor h_alpha = alpha.to(kCPU);
      // std::cout << "training finished" << std::endl;
      float obj = calculate_obj(f_val, h_alpha, h_label);
      rho = calculate_rho(f_val, h_label, h_alpha, Cp, Cn);
      // std::cout << "obj = " << obj << std::endl;
      // std::cout << "rho = " << rho << std::endl;
      break;
    }
  }//  global iter
  /*
     predict
  */
  // alpha obj rho
  // sv_id -> sv_matrix
  // predict_data * svm_atrixT -> pre_kernel_matrix
  // pre_kernel_matrix -> reduce sum -> result
  // std::cout << "predict start" << std::endl;
  auto predict_num_vects = predict_data.dim(0);
  std::vector<int> sv_id_vector;
  std::vector<double> coef_vector;
  //tp.print(alpha);
  Tensor h_alpha = alpha.to(kCPU);
  for (int i = 0; i < num_vects; ++i) {
    h_alpha.mutable_data<double>()[i] *=  h_label.data<int>()[i];
    if(h_alpha.data<double>()[i]!=0){
       sv_id_vector.push_back(i);
       coef_vector.push_back(h_alpha.data<double>()[i]);
    }
  }
  alpha = h_alpha.to(kCUDA);
  Tensor coef = wrap({(long int)coef_vector.size()}, coef_vector.data(), device(kCUDA).dtype(kDouble), true);
  auto sv_size = (long int)sv_id_vector.size();
  Tensor sv_id = wrap({sv_size}, sv_id_vector.data(), device(kCUDA).dtype(kInt32), true);
  // std::cout << "sv_id" << "\n";
  Tensor sv_matrix = full({sv_size, dim_vects}, 0, device(kCUDA).dtype(kFloat));
  get_working_set_from_csr(train_data, sv_id, sv_matrix, sv_size, dim_vects);
  // tp.print(sv_matrix);

  Tensor sv_matrix_row_major = full({sv_size, dim_vects}, 0, device(kCUDA).dtype(kFloat));
  get_working_set_from_csr_row_major(train_data, sv_id, sv_matrix_row_major, sv_size, dim_vects);
  Tensor predict_k_mat_rows_T = full({sv_size, predict_num_vects}, 0, device(kCUDA).dtype(kFloat));
  Tensor predict_k_mat_rows = full({predict_num_vects, sv_size}, 0, device(kCUDA).dtype(kFloat));
  svm_matmul_csr_dense_cuda(predict_data, sv_matrix, predict_k_mat_rows_T);

  // tp.print(predict_k_mat_rows_T);
  predict_k_mat_rows = transpose_matrix(predict_k_mat_rows_T, false);
  // tp.print(predict_k_mat_rows);
  predict_k_mat_rows = contiguous(predict_k_mat_rows);
  // tp.print(predict_k_mat_rows);
  Tensor d_self_dot_sv = full({sv_size}, 0, device(kCUDA).dtype(kFloat));
  get_self_dot(sv_matrix_row_major, d_self_dot_sv);
  Tensor d_self_dot_predict = full({predict_num_vects}, 0, device(kCUDA).dtype(kFloat));
  Tensor data_row_nnz_predict = full({predict_num_vects}, 0, device(kCUDA).dtype(kInt32));
  get_row_nnz(predict_data, data_row_nnz_predict, predict_num_vects);
  get_self_dot_csr(predict_data, d_self_dot_predict, data_row_nnz_predict);
  // RBF_kernel(d_self_dot_predict, d_self_dot_sv, predict_k_mat_rows, predict_num_vects, sv_size, gamma);
  if(param.kernel_type == SvmParam::RBF){
    RBF_kernel(d_self_dot_predict, d_self_dot_sv, predict_k_mat_rows, predict_num_vects, sv_size, gamma);
  }else if(param.kernel_type == SvmParam::POLY){
    poly_kernel(predict_k_mat_rows, gamma, coef0, degree, predict_num_vects*sv_size);
  }
  //tp.print(predict_k_mat_rows);
  // std::cout << "predict finished" << std::endl;
  svm_predict(result, predict_k_mat_rows, coef, predict_num_vects, sv_size);
  // tp.print(result);
#endif
} // function

void svm_impl(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor &result,
         SvmParam param){
    //HICE CHECK svm INPUT

    svm_cuda_csr(train_data, label,predict_data, result, param);
} // function
} //anonymous

HICE_REGISTER_KERNEL(svm_dispatcher, &svm_impl,
                     {kCUDA, kCSR}, // train_data
                     {kCUDA, kDense}, // label
                     {kCUDA, kCSR}, // predict_data
                     {kCUDA, kDense}, // result
);

} // namespace hice

#endif