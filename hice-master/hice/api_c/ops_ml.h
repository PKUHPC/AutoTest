#pragma once

#include "hice/api_c/tensor.h"
#include "hice/api_c/status.h"
#include "hice/api_c/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/// SVM type
typedef enum { 
  C_SVC, 
  EPSILON_SVR, 
  NU_SVC, 
  ONE_CLASS, 
  NU_SVR 
} HI_SvmType;

/// kernel function type
typedef enum { 
  LINEAR, 
  POLY, 
  RBF, 
  SIGMOID, 
  PRECOMPUTED 
} HI_KernelType;

typedef struct {
  HI_SvmType svm_type;
  HI_KernelType kernel_type;
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
} HI_SvmParam;

HICE_API_C HI_Status HI_Svm_Inplace(const HI_Tensor train_data, const HI_Tensor label,
                            const HI_Tensor predict_data, HI_Tensor result,
                            HI_SvmParam param);
HICE_API_C HI_Status HI_Svm(const HI_Tensor train_data, const HI_Tensor label,
                            const HI_Tensor predict_data, HI_Tensor* result,
                            HI_SvmParam param);

#ifdef __cplusplus
}
#endif
