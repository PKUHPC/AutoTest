#include "hice/api_c/ops_ml.h"
#include "hice/api_c/tensor_impl.h"
#include "hice/api_c/error_handle.h"
#include "hice/ml/svm.h"

//inplace
HI_Status HI_Svm_Inplace(const HI_Tensor train_data, const HI_Tensor label,
          const HI_Tensor predict_data, HI_Tensor result,
          HI_SvmParam param) {
  HI_API_BEGIN();
  hice::SvmParam param_in;
  {
    param_in.svm_type = static_cast<hice::SvmParam::SvmType>(param.svm_type);
    param_in.kernel_type = static_cast<hice::SvmParam::KernelType>(param.kernel_type);
    param_in.C =  param.C;
    param_in.gamma = param.gamma;
    param_in.p = param.p;
    param_in.epsilon = param.epsilon;
    param_in.nu = param.nu;
    param_in.probability = param.probability;
    param_in.nr_weight = param.nr_weight;
    param_in.degree = param.degree;
    param_in.coef0 = param.coef0;
    param_in.max_mem_size = param.max_mem_size;
  }
  hice::svm(train_data->tensor_, label->tensor_,
            predict_data->tensor_, result->tensor_,
            param_in);
  HI_API_END();
}

//outplace
HI_Status HI_Svm(const HI_Tensor train_data, const HI_Tensor label,
          const HI_Tensor predict_data, HI_Tensor* result,
          HI_SvmParam param) {
  HI_API_BEGIN();
  hice::SvmParam param_in;
  {
    param_in.svm_type = static_cast<hice::SvmParam::SvmType>(param.svm_type);
    param_in.kernel_type = static_cast<hice::SvmParam::KernelType>(param.kernel_type);
    param_in.C =  param.C;
    param_in.gamma = param.gamma;
    param_in.p = param.p;
    param_in.epsilon = param.epsilon;
    param_in.nu = param.nu;
    param_in.probability = param.probability;
    param_in.nr_weight = param.nr_weight;
    param_in.degree = param.degree;
    param_in.coef0 = param.coef0;
    param_in.max_mem_size = param.max_mem_size;
  }
  *result = new HI_Tensor_Impl{hice::svm(train_data->tensor_, label->tensor_,
                                         predict_data->tensor_, param_in)};
  HI_API_END();
}
