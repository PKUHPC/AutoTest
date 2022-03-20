#ifdef HICE_USE_TVM

#include "hice/nn/separable_conv.h"
#include "hice/tvm/tvm.h"
#include "hice/util/loguru.h"

// #define USE_TVM

#include <string>


#include "hice/tvm/tvm.h"


namespace hice {

Tensor &separable_conv_fwd(const Tensor &input, const Tensor &depth_kernel,
                          const Tensor &point_kernel, 
                          ConstIntArrayRef padding, ConstIntArrayRef stride, 
                          Tensor &output) {
#ifdef USE_TVM
  int64_t in_channel = input.dim(1);
  int64_t out_channel = output.dim(1);
  std::string func_name = std::string("sepconv_") + std::to_string(in_channel) + "_" + 
    std::to_string(out_channel) + "_" + std::to_string(padding[0]) + "_" + std::to_string(stride[0]);

  tvm::runtime::PackedFunc func = TVMHandle::get(func_name.c_str());
  // auto pa = (float*)A.ToDLPack()->dl_tensor.data;
  // auto pb = (float*)B.ToDLPack()->dl_tensor.data;
  // auto pc = (float*)C.ToDLPack()->dl_tensor.data;
  if (func != nullptr) {
    tvm::runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
    tvm::runtime::NDArray dk_tvm = hice::HICETensor_to_NDArray(depth_kernel);
    tvm::runtime::NDArray pk_tvm = hice::HICETensor_to_NDArray(point_kernel);
    tvm::runtime::NDArray out_tvm = hice::HICETensor_to_NDArray(output);
    func(in_tvm, dk_tvm, pk_tvm, out_tvm);    
    if (input.device_type() == kCUDA) {
      TVMSynchronize(kDLGPU, 0, nullptr);
    }
    return output;
  }
  
  HICE_DLOG(INFO) << "Not find TVM kernel:" << func_name 
    << ", Fall back to hice operator.";
#endif  // USE_TVM

  HICE_CHECK_SUPPORTED(false) << "separable_conv_fwd only supportted with TVM.";
  return output;
}

}

#endif