#include "hice/nn/dense.h"

#ifdef HICE_USE_TVM
#include <hice/tvm/tvm.h>
// #define USE_TVM
#endif // HICE_USE_TVM

namespace hice {

Tensor& dense_fwd(const Tensor &input, const Tensor &weight, 
              hice::optional<Tensor> bias, Tensor &output) {
#ifdef USE_TVM
  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t CO = output.dim(1);
  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  
  std::string func_name = "dense_fwd_n" + std::to_string(N) + "_ci" + std::to_string(CI) 
          + "_co" + std::to_string(CO) + + "_" + device_str;

  tvm::runtime::PackedFunc func = TVMHandle::get(func_name.c_str());
  if (func != nullptr) {
    tvm::runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
    tvm::runtime::NDArray krn_tvm = hice::HICETensor_to_NDArray(weight);
    tvm::runtime::NDArray out_tvm = hice::HICETensor_to_NDArray(output);
    func(in_tvm, krn_tvm, out_tvm); 
    if (input.device_type() == kCUDA) {
      TVMSynchronize(kDLGPU, 0, nullptr);
    }
    return output;
  }
  HICE_DLOG(INFO) << "Not find TVM kernel:" << func_name 
    << ", Fall back to hice operator.";
#endif  // USE_TVM

  HICE_CHECK_SUPPORTED(false) << "dense_fwd only supportted with TVM.";
}

std::tuple<Tensor &, Tensor &> dense_bwd(const Tensor &input, const Tensor &weight, const Tensor &grad_output, 
              hice::optional<Tensor> bias, Tensor &grad_input, Tensor &grad_weight) {
#ifdef USE_TVM
  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t CO = grad_output.dim(1);
  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  
  std::string func_name = "dense_bwd_n" + std::to_string(N) + "_ci" + std::to_string(CI) 
          + "_co" + std::to_string(CO) + + "_" + device_str;

  tvm::runtime::PackedFunc func = TVMHandle::get(func_name.c_str());
  if (func != nullptr) {
    tvm::runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
    tvm::runtime::NDArray krn_tvm = hice::HICETensor_to_NDArray(weight);
    tvm::runtime::NDArray grad_out_tvm = hice::HICETensor_to_NDArray(grad_output);
    tvm::runtime::NDArray grad_in_tvm = hice::HICETensor_to_NDArray(grad_input);
    tvm::runtime::NDArray grad_weight_tvm = hice::HICETensor_to_NDArray(grad_weight);
    func(in_tvm, krn_tvm, grad_out_tvm, grad_in_tvm, grad_weight_tvm); 
    if (input.device_type() == kCUDA) {
      TVMSynchronize(kDLGPU, 0, nullptr);
    }
    return std::tuple<Tensor&, Tensor&>{grad_input, grad_weight};
  }
  HICE_DLOG(INFO) << "Not find TVM kernel:" << func_name 
    << ", Fall back to hice operator.";
#endif  // USE_TVM

  HICE_CHECK_SUPPORTED(false) << "dense_bwd only supportted with TVM.";
  return std::tuple<Tensor&, Tensor&>{grad_input, grad_weight};
  
}


} // namespace hice
