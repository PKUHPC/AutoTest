#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/ml/svm.h"
#include "hice/ml/cpu/svm_kernel.h"
#include "hice/basic/transpose.h"
#include "hice/basic/reshape.h"
#include "hice/basic/copy.h"
#include "hice/math/matmul.h"


namespace hice {


namespace{
void svm_cpu(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor &result,
         SvmParam param){
  //  std::cout << "=========Kernel: svm_cpu==========" << std::endl;
   TensorPrinter tp;
   auto num_vects = train_data.dim(0);
   auto dim_vects = train_data.dim(1);
   double rho = 0.0;
   // tp.print(train_data);
   Tensor train_data_T = transpose_matrix(train_data, false);
   // tp.print(train_data_T);
   train_data_T = contiguous(train_data_T);
   // tp.print(train_data_T);
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

   Tensor alpha = full({num_vects}, 0, device(kCPU).dtype(kDouble));
   Tensor k_mat_rows = full({ws_size, num_vects},
          0, device(kCPU).dtype(kFloat));
   Tensor k_mat_rows_first_half = wrap({half_ws_size, num_vects},
          k_mat_rows.mutable_data<float>(),
           device(kCPU).dtype(kFloat), false);
   Tensor k_mat_rows_last_half = wrap({half_ws_size, num_vects},
           k_mat_rows.mutable_data<float>() + k_mat_rows.size() / 2,
           device(kCPU).dtype(kFloat), false);

   Tensor working_set({1, ws_size}, device(kCPU).dtype(kInt32));
   Tensor working_set_first_half =
       wrap({1, half_ws_size}, working_set.mutable_data<int>(),
            device(kCPU).dtype(kInt32), false);
   Tensor working_set_last_half = wrap(
       {1, half_ws_size}, working_set.mutable_data<int>() + working_set.size() / 2,
       device(kCPU).dtype(kInt32), false);

   Tensor self_dot = full({num_vects, 1}, 0, device(kCPU).dtype(kFloat));
   get_self_dot_cpu(train_data, self_dot);
   Tensor data_rows =
       full({ws_size, dim_vects}, 0, device(kCPU).dtype(kFloat));
   Tensor data_rows_half =
       full({half_ws_size, dim_vects}, 0, device(kCPU).dtype(kFloat));
   Tensor f_val({1, num_vects}, device(kCPU).dtype(kDouble));
   Tensor f_idx({1, num_vects}, device(kCPU).dtype(kInt32));
   Tensor f_idx2sort = full({1, num_vects}, 0, device(kCPU).dtype(kInt32));
   Tensor f_val2sort = full({1, num_vects}, 0, device(kCPU).dtype(kDouble));
   Tensor alpha_diff = full({1, ws_size}, 0, device(kCPU).dtype(kDouble));
   Tensor diff = full({1, 2}, 0, device(kCPU).dtype(kDouble));
   Tensor kmat_diag = full({1, num_vects}, 1, device(kCPU).dtype(kFloat));
   for (int i = 0; i < num_vects; ++i) {
     f_idx.mutable_data<int>()[i] = i;
     f_val.mutable_data<double>()[i] = -1 * label.data<int>()[i];
   }
   Tensor ws_indicator({num_vects, 1}, device(kCPU).dtype(kInt32));
   int *ws_ind = ws_indicator.mutable_data<int>();
   int *w_s_f_h = working_set_first_half.mutable_data<int>();
   int ws_ind_size = sizeof(int) * num_vects;

   long long local_iter = 0;
   int g_max_iter = 1000;
   int out_max_iter = -1;
   int max_iter = std::max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
  //  std::cout << "traing start" << std::endl;
   for (int iter = 0;; ++iter) // global _ iter
   {
     copy(f_idx, f_idx2sort);
     copy(f_val, f_val2sort);
     sort_f_cpu(f_val2sort, f_idx2sort);
     memset(ws_ind, 0, ws_ind_size);
     if (0 == iter) {
       select_working_set_cpu(ws_indicator, f_idx2sort, label, alpha, Cp, Cn, working_set);
       get_working_set_cpu(train_data, working_set, data_rows, ws_size, dim_vects);
       // svm_matmul_cpu(data_rows, train_data_T, k_mat_rows);
       matmul(data_rows, train_data_T, k_mat_rows, kNoTrans, kNoTrans);
       // tp.print(k_mat_rows);
       // RBF_kernel_cpu(working_set, self_dot, k_mat_rows, ws_size, num_vects,gamma);
       if(param.kernel_type == SvmParam::RBF){
         RBF_kernel_cpu(working_set, self_dot, k_mat_rows, ws_size, num_vects, gamma);
       }else if(param.kernel_type == SvmParam::POLY){
         poly_kernel_cpu(k_mat_rows, gamma, coef0, degree, ws_size*num_vects);
       }
     }
     else {
       copy(k_mat_rows_last_half, k_mat_rows_first_half);
       copy(working_set_last_half, working_set_first_half);
 #pragma omp parallel for schedule(guided)
       for (int i = 0; i < half_ws_size; ++i) {
         ws_ind[w_s_f_h[i]] = 1; //
       }
       select_working_set_cpu(ws_indicator, f_idx2sort, label, alpha, Cp, Cn, working_set_last_half);
       get_working_set_cpu(train_data, working_set_last_half, data_rows_half,
                           half_ws_size, dim_vects);
       // svm_matmul_cpu(data_rows_half, train_data_T, k_mat_rows_last_half);
       matmul(data_rows_half, train_data_T, k_mat_rows_last_half,  kNoTrans, kNoTrans);
       // tp.print(k_mat_rows_last_half);
       // RBF_kernel_cpu(working_set_last_half, self_dot, k_mat_rows_last_half,
                      // half_ws_size, num_vects, gamma);
       if(param.kernel_type == SvmParam::RBF){
         RBF_kernel_cpu(working_set_last_half, self_dot, k_mat_rows_last_half,
                     half_ws_size, num_vects, gamma);
       }else if(param.kernel_type == SvmParam::POLY){
         poly_kernel_cpu(k_mat_rows_last_half, gamma, coef0, degree, half_ws_size*num_vects);
       }
     }
     c_smo_solve_cpu(label, f_val, alpha, alpha_diff, working_set, Cp, Cn,
                     k_mat_rows, kmat_diag, num_vects, 0.001, diff, max_iter);
     update_f_cpu(f_val, alpha_diff, k_mat_rows, num_vects);
     const double *diff_data = diff.data<double>();
     local_iter += diff_data[1];

     // track unchanged diff
     if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
       same_local_diff_cnt++;
     } else {
       same_local_diff_cnt = 0;
       previous_local_diff = diff_data[0];
     }

     // track unchanged swapping diff
     if (fabs(diff_data[0] - second_last_local_diff) < eps * 0.001) {
       swap_local_diff_cnt++;
     } else {
       swap_local_diff_cnt = 0;
     }
     second_last_local_diff = last_local_diff;
     last_local_diff = diff_data[0];

     if (iter % 100 == 0)
       // std::cout << "global iter = " << iter
       //           << ", total iter = " << local_iter
       //           << ", diff = " << diff_data[0] << std::endl;
     if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) ||
         diff_data[0] < eps || (out_max_iter != -1) && (iter == out_max_iter) ||
         (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
       //
       // std::cout << "global iter = " << iter
       //           << ", total iter = " << local_iter
       //           << ", diff = " << diff_data[0] << std::endl;
      //  std::cout << "training finished" << std::endl;
       float obj = calculate_obj(f_val, alpha, label);
       rho = calculate_rho(f_val, label, alpha, Cp, Cn);
      //  std::cout << "obj = " << obj << std::endl;
      //  std::cout << "rho = " << rho << std::endl;
       break;
     }
   } // global iter

   /**
     * predict
   */
  //  std::cout << "predict start" << std::endl;
   auto predict_num_vects = predict_data.dim(0);
   std::vector<int> sv_id_vector;
   std::vector<double> coef_vector;
   // tp.print(alpha);
   for (int i = 0; i < num_vects; ++i) {
     alpha.mutable_data<double>()[i] *=  label.data<int>()[i];
     if(alpha.data<double>()[i]!=0){
        sv_id_vector.push_back(i);
        coef_vector.push_back(alpha.data<double>()[i]);
     }
   }
   // tp.print(alpha);
   auto sv_size = (long int)sv_id_vector.size();
   Tensor sv_id = full({sv_size}, 0, device(kCPU).dtype(kInt32));
   Tensor coef = full({sv_size}, 0, device(kCPU).dtype(kDouble));
   for(int i = 0; i < sv_size; i++){
      sv_id.mutable_data<int>()[i] = sv_id_vector[i];
      coef.mutable_data<double>()[i] = coef_vector[i];
   }
   // tp.print(sv_id);
   Tensor sv_matrix = full({sv_size, dim_vects}, 0, device(kCPU).dtype(kFloat));
   get_working_set_cpu(train_data, sv_id, sv_matrix, sv_size, dim_vects);
   // tp.print(sv_matrix);
   Tensor sv_matrix_T = transpose_matrix(sv_matrix, false);
   // tp.print(sv_matrix_T);
   sv_matrix_T = contiguous(sv_matrix_T);
   // tp.print(sv_matrix_T);
   Tensor predict_k_mat_rows = full({predict_num_vects, sv_size}, 0, device(kCPU).dtype(kFloat));
   // svm_matmul_cpu(predict_data, sv_matrix_T, predict_k_mat_rows);
   matmul(predict_data, sv_matrix_T, predict_k_mat_rows, kNoTrans, kNoTrans);
   // tp.print(predict_k_mat_rows);
   Tensor self_dot_sv = full({sv_size}, 0, device(kCPU).dtype(kFloat));
   get_self_dot_cpu(sv_matrix, self_dot_sv);
   Tensor self_dot_predict = full({predict_num_vects}, 0, device(kCPU).dtype(kFloat));
   get_self_dot_cpu(predict_data, self_dot_predict);
   // RBF_kernel_cpu(self_dot_predict, self_dot_sv, predict_k_mat_rows, predict_num_vects, sv_size, gamma);
   if(param.kernel_type == SvmParam::RBF){
     RBF_kernel_cpu(self_dot_predict, self_dot_sv, predict_k_mat_rows, predict_num_vects, sv_size, gamma);
   }else if(param.kernel_type == SvmParam::POLY){
     poly_kernel_cpu(predict_k_mat_rows, gamma, coef0, degree, predict_num_vects*sv_size);
   }
   // tp.print(predict_k_mat_rows);
   svm_predict_cpu(result, predict_k_mat_rows, coef, predict_num_vects, sv_size);
  //  std::cout << "predict finished" << std::endl;
  //  tp.print(result);
} // function


void svm_impl(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor &result,
         SvmParam param){
    svm_cpu(train_data, label,predict_data, result, param);
} // function
}
HICE_REGISTER_KERNEL(svm_dispatcher, &svm_impl,
                     {kCPU, kDense}, // train_data
                     {kCPU, kDense}, // label
                     {kCPU, kDense}, // predict_data
                     {kCPU, kDense}, // result
);

} // namespace hice
