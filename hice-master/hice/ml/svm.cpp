// #if 0

#include "hice/ml/svm.h"
namespace hice {

int get_working_set_size(int n_instances, int n_features) {
  size_t max_mem_size = static_cast<size_t>(8192) << 20;  //8
  size_t free_mem = max_mem_size - 2 * (n_instances * n_features) * sizeof(float); // dense
  int ws_size = std::min(max2power(n_instances), (int)std::min(max2power(free_mem / sizeof(float) /
                                                 (n_instances + n_features)), size_t(1024)));
  return ws_size;
}
double calculate_rho(const Tensor& f_val, const Tensor& y, const Tensor& alpha, double Cp,
                     double Cn) {
  int n_free = 0;
  double sum_free = 0;
  double up_value = INFINITY;
  double low_value = -INFINITY;
  const double *f_val_data = f_val.data<double>();
  const int *y_data = y.data<int>();
  const double *alpha_data = alpha.data<double>();
  for (int i = 0; i < alpha.size(); ++i) {
    if (is_free(alpha_data[i], y_data[i], Cp, Cn)) {
      n_free++;
      sum_free += f_val_data[i];
    }
    if (is_I_up(alpha_data[i], y_data[i], Cp, Cn))
      up_value = std::min(up_value, f_val_data[i]);
    if (is_I_low(alpha_data[i], y_data[i], Cp, Cn))
      low_value = std::max(low_value, f_val_data[i]);
  }
  return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}
double calculate_obj(const Tensor& f_val, const Tensor& alpha, const Tensor& y) {
  int n_instances = f_val.size();
  double obj = 0;
  const double *f_val_data = f_val.data<double>();
  const double *alpha_data = alpha.data<double>();
  const int *y_data = y.data<int>();
  for (int i = 0; i < n_instances; ++i) {
    obj += alpha_data[i] -
           (f_val_data[i] + y_data[i]) * alpha_data[i] * y_data[i] / 2;
  }
  return -obj;
}

HICE_DEFINE_DISPATCHER(svm_dispatcher);

//inplace
void svm(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor& result,
         SvmParam param){
     // HICE CHECK SVM INPUT
     ScalarType sc_train_type = train_data.scalar_type();
     ScalarType sc_predict_type = predict_data.scalar_type();
     ScalarType sc_label_type = label.scalar_type();
     ScalarType sc_result_type = result.scalar_type();
     const int num_vects = train_data.dim(0);
     const int num_label = label.dim(0);
     const int predict_num_vects = predict_data.dim(0);
     const int num_result = result.dim(0);
     HICE_CHECK_TYPE_MATCH(sc_train_type == kFloat)
         << "type of train data to svm must be float";
     HICE_CHECK_TYPE_MATCH(sc_train_type == sc_predict_type)
         << "types of train and predict data to svm must be equal";
     HICE_CHECK_TYPE_MATCH(sc_label_type == kInt32)
         << "types of label to svm must be int";
     HICE_CHECK_TYPE_MATCH(sc_result_type == kInt32)
         << "types of result to svm must be int";
     HICE_CHECK_DIMS_MATCH(num_vects == num_label)
         << "num of train_data and label to svm must be equal";
     HICE_CHECK_DIMS_MATCH(predict_num_vects == num_result)
         << "num of predict_data and result to svm must be equal";

     svm_dispatcher(train_data, label, predict_data, result, param);
}

//outplace
Tensor svm(const Tensor& train_data, const Tensor& label,
           const Tensor& predict_data, SvmParam param){
     // HICE CHECK SVM INPUT
     ScalarType sc_train_type = train_data.scalar_type();
     ScalarType sc_predict_type = predict_data.scalar_type();
     ScalarType sc_label_type = label.scalar_type();
     const int num_vects = train_data.dim(0);
     const int num_label = label.dim(0);
     const int predict_num_vects = predict_data.dim(0);
     const int num_result = predict_num_vects;
     Tensor result({num_result}, device(label.device()).dtype(label.data_type()));
     ScalarType sc_result_type = result.scalar_type();
     HICE_CHECK_TYPE_MATCH(sc_train_type == kFloat)
         << "type of train data to svm must be float";
     HICE_CHECK_TYPE_MATCH(sc_train_type == sc_predict_type)
         << "types of train and predict data to svm must be equal";
     HICE_CHECK_TYPE_MATCH(sc_label_type == kInt32)
         << "types of label to svm must be int";
     HICE_CHECK_TYPE_MATCH(sc_result_type == kInt32)
         << "types of result to svm must be int";
     HICE_CHECK_DIMS_MATCH(num_vects == num_label)
         << "num of train_data and label to svm must be equal";
     HICE_CHECK_DIMS_MATCH(predict_num_vects == num_result)
         << "num of predict_data and result to svm must be equal";
     svm_dispatcher(train_data, label, predict_data, result, param);

     return result;
}

} // namespace hice

// #endif