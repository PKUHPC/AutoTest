#pragma once

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"
#include "hice/core/expression_util.h"


namespace hice {


class SvmParam {
public:
  SvmParam() {
    svm_type = C_SVC;
    kernel_type = RBF;
    C =  4;
    gamma = 0.5;
    p = 0.1f;
    epsilon = 0.001f;
    nu = 0.5;
    probability = false;
    nr_weight = 0;
    degree = 3;
    coef0 = 0;
    max_mem_size = static_cast<size_t>(8192) << 20;
  }
  /// SVM type
  enum SvmType { C_SVC, EPSILON_SVR, NU_SVC, ONE_CLASS, NU_SVR };
  /// kernel function type
  enum KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };
  SvmType svm_type;
  KernelType kernel_type;
  /// regularization parameter
  double C;
  /// for RBF kernel
  double gamma;
  /// for regression
  double p;
  /// for \f$\nu\f$-SVM
  double nu;
  /// stopping criteria
  double epsilon;
  /// degree for polynomial kernel
  int degree;
  /// for polynomial/sigmoid kernel
  double coef0;
  /// for SVC
  int nr_weight;
  /// for SVC
  int *weight_label;
  /// for SVC
  double *weight;
  /// do probability estimates
  int probability;
  /// maximum memory size
  size_t max_mem_size;
};

int get_working_set_size(int n_instances, int n_features);

template <typename T> inline T max2power(T n) {
   return T(pow(2, floor(log2f(float(n)))));
}
inline bool is_I_up(double a, double y, double Cp, double Cn) {
  return (y > 0 && a < Cp) || (y < 0 && a > 0);
}

inline bool is_I_low(double a, double y, double Cp, double Cn) {
  return (y > 0 && a > 0) || (y < 0 && a < Cn);
}

inline bool is_free(double a, double y, double Cp, double Cn) {
  return a > 0 && (y > 0 ? a < Cp : a < Cn);
}
double calculate_rho(const Tensor& f_val, const Tensor& y, const Tensor& alpha, double Cp,
                     double Cn);
double calculate_obj(const Tensor& f_val, const Tensor& alpha, const Tensor& y);

// Dispatcher
using svm_kernel_fn_type = void (*)
                        (const Tensor& train_data, const Tensor& label,
                         const Tensor& predict_data, Tensor &result,
                         SvmParam param);
HICE_DECLARE_DISPATCHER(svm_dispatcher, svm_kernel_fn_type);

// Operators
//inplace
HICE_API void svm(const Tensor& train_data, const Tensor& label,
         const Tensor& predict_data, Tensor& result,
         SvmParam param);
//outplace
HICE_API Tensor svm(const Tensor& train_data, const Tensor& label,
                   const Tensor& predict_data, SvmParam param);

} // namespace hice
