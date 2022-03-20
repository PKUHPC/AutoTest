#include <hice/tvm/operators.h>
#include <hice/tvm/tvm.h>
#include <hice/tvm/py_apis.h>
#include <hice/util/dlpack_wrapper.h>
#include <hice/util/benchmark.h>

#include <sys/time.h>

namespace hice {

namespace {
  
inline void SetPargs(HicePyObject& pArgsTuple, HicePyObject& pArgsDict, std::string& func_name) {
  HICE_PyTuple_SetItem(pArgsTuple, 0, pArgsDict);
  HICE_PyTuple_SetItem(pArgsTuple, 1, HICE_Py_BuildValue("s", func_name.c_str()));
  HICE_PyTuple_SetItem(pArgsTuple, 2, HICE_Py_BuildValue("s", TVMLibConfig::prefix().c_str()));
  HICE_PyTuple_SetItem(pArgsTuple, 3, HICE_Py_BuildValue("i", TVMLibConfig::n_search_trails()));
}

} // namespace anonymous

bool pooling_avg_fwd_tvm(const Tensor& input, ConstIntArrayRef kernel,
                            ConstIntArrayRef stride, ConstIntArrayRef padding, 
                            Tensor& output) {
  // HICE_DLOG(INFO) << "enter pooling_avg_fwd_tvm";

  int64_t N = input.dim(0);
  int64_t C = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);
  int64_t ksize = kernel[0];
  int64_t strid = stride[0];
  int64_t pad = padding[0];
  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";

  std::string func_name = "pool_avg_fwd_n" + std::to_string(N) + "_c" + std::to_string(C) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
  
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("pool_avg_fwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "C", HICE_Py_BuildValue("i", C));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  // HICE_DLOG(INFO) << "got tvm_func: " << func_name << " from .so";
  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_output = HICETensor_to_DLManagedTensor(output);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray out_tvm = runtime::NDArray::FromDLPack(&dlm_output);
  
  // HICE_DLOG(INFO) << "begin execute tvm kernel...";
  func(in_tvm, out_tvm); 
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool pooling_avg_bwd_tvm(const Tensor& input, const Tensor& output,
                            const Tensor& grad_output,
                            ConstIntArrayRef kernel, ConstIntArrayRef stride,
                            ConstIntArrayRef padding, Tensor& grad_input) {
  int64_t N = input.dim(0);
  int64_t C = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);
  int64_t HO = grad_output.dim(2);
  int64_t WO = grad_output.dim(3);
  int64_t ksize = kernel[0];
  int64_t strid = stride[0];
  int64_t pad = padding[0];
  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";

  std::string func_name = "pool_avg_bwd_n" + std::to_string(N) + "_c" + std::to_string(C) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_ho" + std::to_string(HO) + "_wo" + std::to_string(WO) 
          + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;

  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("pool_avg_bwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "C", HICE_Py_BuildValue("i", C));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HICE_PyDict_SetItemString(pArgsDict, "HO", HICE_Py_BuildValue("i", HO));
    HICE_PyDict_SetItemString(pArgsDict, "WO", HICE_Py_BuildValue("i", WO));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_grad_out = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_in = HICETensor_to_DLManagedTensor(grad_input);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray grad_out_tvm = runtime::NDArray::FromDLPack(&dlm_grad_out);
  tvm::runtime::NDArray grad_in_tvm = runtime::NDArray::FromDLPack(&dlm_grad_in);
  func(in_tvm, grad_out_tvm, grad_in_tvm); 
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}


bool conv_fwd_tvm(const Tensor &input, const Tensor &weight,
                      ConstIntArrayRef padding,
                      ConstIntArrayRef stride, ConstIntArrayRef dilation,
                      Tensor &output) {

  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);
  int64_t CO = output.dim(1);
  int64_t ksize = weight.dim(2);
  int64_t strid = stride[0];
  int64_t pad = padding[0];

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "conv_fwd_n" + std::to_string(N) + "_c" + std::to_string(CI) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_co" + std::to_string(CO) + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("conv_fwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_weight = HICETensor_to_DLManagedTensor(weight);
  DLManagedTensor dlm_output = HICETensor_to_DLManagedTensor(output);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray krn_tvm = runtime::NDArray::FromDLPack(&dlm_weight);
  tvm::runtime::NDArray out_tvm = runtime::NDArray::FromDLPack(&dlm_output);
  func(in_tvm, krn_tvm, out_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool conv_bwd_tvm(const Tensor &input, const Tensor &weight, 
                      const Tensor &grad_output,
                      ConstIntArrayRef padding, ConstIntArrayRef stride,
                      ConstIntArrayRef dilation,
                      Tensor& grad_input, Tensor& grad_weight) {

  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t HI = input.dim(2);
  int64_t WI = input.dim(3);
  int64_t CO = grad_output.dim(1);
  int64_t HO = grad_output.dim(2);
  int64_t WO = grad_output.dim(3);
  int64_t ksize = weight.dim(2);
  int64_t strid = stride[0];
  int64_t pad = padding[0];

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "conv_bwd_n" + std::to_string(N) + "_ci" + std::to_string(CI) 
          + "_hi" + std::to_string(HI) + "_wi" + std::to_string(WI) 
          + "_co" + std::to_string(CO) 
          + "_ho" + std::to_string(HO) + "_wo" + std::to_string(WO)
          + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("conv_bwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "HI", HICE_Py_BuildValue("i", HI));
    HICE_PyDict_SetItemString(pArgsDict, "WI", HICE_Py_BuildValue("i", WI));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HICE_PyDict_SetItemString(pArgsDict, "HO", HICE_Py_BuildValue("i", HO));
    HICE_PyDict_SetItemString(pArgsDict, "WO", HICE_Py_BuildValue("i", WO));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_weight = HICETensor_to_DLManagedTensor(weight);
  DLManagedTensor dlm_grad_output = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_input = HICETensor_to_DLManagedTensor(grad_input);
  DLManagedTensor dlm_grad_weight = HICETensor_to_DLManagedTensor(grad_weight);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray wgt_tvm = runtime::NDArray::FromDLPack(&dlm_weight);
  tvm::runtime::NDArray grad_out_tvm = runtime::NDArray::FromDLPack(&dlm_grad_output);
  tvm::runtime::NDArray grad_in_tvm = runtime::NDArray::FromDLPack(&dlm_grad_input);
  tvm::runtime::NDArray grad_wgt_tvm = runtime::NDArray::FromDLPack(&dlm_grad_weight);
  func(in_tvm, wgt_tvm, grad_out_tvm, grad_in_tvm, grad_wgt_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool conv_bwd_input_tvm(const Tensor &input, const Tensor &weight, 
                      const Tensor &grad_output,
                      ConstIntArrayRef padding, ConstIntArrayRef stride,
                      ConstIntArrayRef dilation,
                      Tensor& grad_input) {

  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t HI = input.dim(2);
  int64_t WI = input.dim(3);
  int64_t CO = grad_output.dim(1);
  int64_t HO = grad_output.dim(2);
  int64_t WO = grad_output.dim(3);
  int64_t ksize = weight.dim(2);
  int64_t strid = stride[0];
  int64_t pad = padding[0];

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "conv_bwd_input_n" + std::to_string(N) + "_ci" + std::to_string(CI) 
          + "_hi" + std::to_string(HI) + "_wi" + std::to_string(WI) 
          + "_co" + std::to_string(CO) 
          + "_ho" + std::to_string(HO) + "_wo" + std::to_string(WO)
          + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("conv_bwd_input_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "HI", HICE_Py_BuildValue("i", HI));
    HICE_PyDict_SetItemString(pArgsDict, "WI", HICE_Py_BuildValue("i", WI));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HICE_PyDict_SetItemString(pArgsDict, "HO", HICE_Py_BuildValue("i", HO));
    HICE_PyDict_SetItemString(pArgsDict, "WO", HICE_Py_BuildValue("i", WO));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_weight = HICETensor_to_DLManagedTensor(weight);
  DLManagedTensor dlm_grad_output = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_input = HICETensor_to_DLManagedTensor(grad_input);
  tvm::runtime::NDArray wgt_tvm = runtime::NDArray::FromDLPack(&dlm_weight);
  tvm::runtime::NDArray grad_out_tvm = runtime::NDArray::FromDLPack(&dlm_grad_output);
  tvm::runtime::NDArray grad_in_tvm = runtime::NDArray::FromDLPack(&dlm_grad_input);
  func(wgt_tvm, grad_out_tvm, grad_in_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool conv_bwd_weight_tvm(const Tensor &input, const Tensor &weight, 
                      const Tensor &grad_output,
                      ConstIntArrayRef padding, ConstIntArrayRef stride,
                      ConstIntArrayRef dilation,
                      Tensor& grad_weight) {

  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t HI = input.dim(2);
  int64_t WI = input.dim(3);
  int64_t CO = grad_output.dim(1);
  int64_t HO = grad_output.dim(2);
  int64_t WO = grad_output.dim(3);
  int64_t ksize = weight.dim(2);
  int64_t strid = stride[0];
  int64_t pad = padding[0];

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "conv_bwd_weight_n" + std::to_string(N) + "_ci" + std::to_string(CI) 
          + "_hi" + std::to_string(HI) + "_wi" + std::to_string(WI) 
          + "_co" + std::to_string(CO) 
          + "_ho" + std::to_string(HO) + "_wo" + std::to_string(WO)
          + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("conv_bwd_weight_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "HI", HICE_Py_BuildValue("i", HI));
    HICE_PyDict_SetItemString(pArgsDict, "WI", HICE_Py_BuildValue("i", WI));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HICE_PyDict_SetItemString(pArgsDict, "HO", HICE_Py_BuildValue("i", HO));
    HICE_PyDict_SetItemString(pArgsDict, "WO", HICE_Py_BuildValue("i", WO));
    HICE_PyDict_SetItemString(pArgsDict, "ksize", HICE_Py_BuildValue("i", ksize));
    HICE_PyDict_SetItemString(pArgsDict, "stride", HICE_Py_BuildValue("i", strid));
    HICE_PyDict_SetItemString(pArgsDict, "padding", HICE_Py_BuildValue("i", pad));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_grad_output = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_weight = HICETensor_to_DLManagedTensor(grad_weight);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray grad_out_tvm = runtime::NDArray::FromDLPack(&dlm_grad_output);
  tvm::runtime::NDArray grad_wgt_tvm = runtime::NDArray::FromDLPack(&dlm_grad_weight);
  func(in_tvm, grad_out_tvm, grad_wgt_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}


bool dense_fwd_tvm(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output) {
  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t CO = weight.dim(0);

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "dense_fwd_n" + std::to_string(N) + "_c" + std::to_string(CI) 
          + "_co" + std::to_string(CO) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("dense_fwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_weight = HICETensor_to_DLManagedTensor(weight);
  DLManagedTensor dlm_bias = HICETensor_to_DLManagedTensor(bias);
  DLManagedTensor dlm_output = HICETensor_to_DLManagedTensor(output);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray wgt_tvm = runtime::NDArray::FromDLPack(&dlm_weight);
  tvm::runtime::NDArray bias_tvm = runtime::NDArray::FromDLPack(&dlm_bias);
  tvm::runtime::NDArray out_tvm = runtime::NDArray::FromDLPack(&dlm_output);
  func(in_tvm, wgt_tvm, bias_tvm, out_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}
                              
bool dense_bwd_tvm(const Tensor &input, const Tensor &weight, const Tensor &grad_output, 
                      Tensor &grad_input, Tensor &grad_weight, Tensor &grad_bias) {
  int64_t N = input.dim(0);
  int64_t CI = input.dim(1);
  int64_t CO = weight.dim(0);

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "dense_bwd_n" + std::to_string(N) + "_c" + std::to_string(CI) 
          + "_co" + std::to_string(CO)
          + "_" + device_str;
  
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("dense_bwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "CI", HICE_Py_BuildValue("i", CI));
    HICE_PyDict_SetItemString(pArgsDict, "CO", HICE_Py_BuildValue("i", CO));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_weight = HICETensor_to_DLManagedTensor(weight);
  DLManagedTensor dlm_grad_output = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_input = HICETensor_to_DLManagedTensor(grad_input);
  DLManagedTensor dlm_grad_weight = HICETensor_to_DLManagedTensor(grad_weight);
  DLManagedTensor dlm_grad_bias = HICETensor_to_DLManagedTensor(grad_bias);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray wgt_tvm = runtime::NDArray::FromDLPack(&dlm_weight);
  tvm::runtime::NDArray grad_output_tvm = runtime::NDArray::FromDLPack(&dlm_grad_output);
  tvm::runtime::NDArray grad_input_tvm = runtime::NDArray::FromDLPack(&dlm_grad_input);
  tvm::runtime::NDArray grad_weight_tvm = runtime::NDArray::FromDLPack(&dlm_grad_weight);
  tvm::runtime::NDArray grad_bias_tvm = runtime::NDArray::FromDLPack(&dlm_grad_bias);
  func(in_tvm, wgt_tvm, grad_output_tvm, grad_bias_tvm, grad_input_tvm, grad_weight_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool batch_norm_fwd_tvm(const Tensor &input, const Tensor &scale, const Tensor &bias,
                          const Tensor &running_mean, const Tensor &running_var, 
                          const Tensor &momentum, const Tensor &eps, 
                          Tensor &output, Tensor& saved_mean, Tensor& saved_var) {
  int64_t N = input.dim(0);
  int64_t C = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "batch_norm_fwd_n" + std::to_string(N) + "_c" + std::to_string(C) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("batch_norm_fwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "C", HICE_Py_BuildValue("i", C));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_scale = HICETensor_to_DLManagedTensor(scale);
  DLManagedTensor dlm_bias = HICETensor_to_DLManagedTensor(bias);
  DLManagedTensor dlm_running_mean = HICETensor_to_DLManagedTensor(running_mean);
  DLManagedTensor dlm_running_var = HICETensor_to_DLManagedTensor(running_var);
  DLManagedTensor dlm_momentum = HICETensor_to_DLManagedTensor(momentum);
  DLManagedTensor dlm_eps = HICETensor_to_DLManagedTensor(eps);
  DLManagedTensor dlm_output = HICETensor_to_DLManagedTensor(output);
  DLManagedTensor dlm_saved_mean = HICETensor_to_DLManagedTensor(saved_mean);
  DLManagedTensor dlm_saved_var = HICETensor_to_DLManagedTensor(saved_var);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray scl_tvm = runtime::NDArray::FromDLPack(&dlm_scale);
  tvm::runtime::NDArray bias_tvm = runtime::NDArray::FromDLPack(&dlm_bias);
  tvm::runtime::NDArray rmean_tvm = runtime::NDArray::FromDLPack(&dlm_running_mean);
  tvm::runtime::NDArray rvar_tvm = runtime::NDArray::FromDLPack(&dlm_running_var);
  tvm::runtime::NDArray momentum_tvm = runtime::NDArray::FromDLPack(&dlm_momentum);
  tvm::runtime::NDArray eps_tvm = runtime::NDArray::FromDLPack(&dlm_eps);
  tvm::runtime::NDArray out_tvm = runtime::NDArray::FromDLPack(&dlm_output);
  tvm::runtime::NDArray smean_tvm = runtime::NDArray::FromDLPack(&dlm_saved_mean);
  tvm::runtime::NDArray svar_tvm = runtime::NDArray::FromDLPack(&dlm_saved_var);
  func(in_tvm, scl_tvm, bias_tvm, rmean_tvm, rvar_tvm, momentum_tvm, eps_tvm, out_tvm, smean_tvm, svar_tvm, rmean_tvm, rvar_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool batch_norm_bwd_tvm(const Tensor &input, const Tensor &scale, const Tensor &saved_mean,
                          const Tensor &saved_rvars, const Tensor &eps, 
                          const Tensor &grad_output,
                          Tensor &grad_input, Tensor& grad_scale, Tensor& grad_bias) {
  int64_t N = input.dim(0);
  int64_t C = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "batch_norm_bwd_n" + std::to_string(N) + "_c" + std::to_string(C) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_" + device_str;
  
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("batch_norm_bwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "C", HICE_Py_BuildValue("i", C));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_scale = HICETensor_to_DLManagedTensor(scale);
  DLManagedTensor dlm_saved_mean = HICETensor_to_DLManagedTensor(saved_mean);
  DLManagedTensor dlm_saved_rvars = HICETensor_to_DLManagedTensor(saved_rvars);
  DLManagedTensor dlm_eps = HICETensor_to_DLManagedTensor(eps);
  DLManagedTensor dlm_grad_output = HICETensor_to_DLManagedTensor(grad_output);
  DLManagedTensor dlm_grad_input = HICETensor_to_DLManagedTensor(grad_input);
  DLManagedTensor dlm_grad_scale = HICETensor_to_DLManagedTensor(grad_scale);
  DLManagedTensor dlm_grad_bias = HICETensor_to_DLManagedTensor(grad_bias);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray scl_tvm = runtime::NDArray::FromDLPack(&dlm_scale);
  tvm::runtime::NDArray smean_tvm = runtime::NDArray::FromDLPack(&dlm_saved_mean);
  tvm::runtime::NDArray srvars_tvm = runtime::NDArray::FromDLPack(&dlm_saved_rvars);
  tvm::runtime::NDArray eps_tvm = runtime::NDArray::FromDLPack(&dlm_eps);
  tvm::runtime::NDArray grad_out_tvm = runtime::NDArray::FromDLPack(&dlm_grad_output);
  tvm::runtime::NDArray grad_in_tvm = runtime::NDArray::FromDLPack(&dlm_grad_input);
  tvm::runtime::NDArray grad_scale_tvm = runtime::NDArray::FromDLPack(&dlm_grad_scale);
  tvm::runtime::NDArray grad_bias_tvm = runtime::NDArray::FromDLPack(&dlm_grad_bias);
  func(in_tvm, scl_tvm, smean_tvm, srvars_tvm, eps_tvm, grad_out_tvm, grad_in_tvm, grad_scale_tvm, grad_bias_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}

bool relu_fwd_tvm(const Tensor &input, Tensor &output) {
  int64_t N = input.dim(0);
  int64_t C = input.dim(1);
  int64_t H = input.dim(2);
  int64_t W = input.dim(3);

  std::string device_str = input.device_type() == kCUDA ? "cuda" : "cpu";
  std::string func_name = "relu_fwd_n" + std::to_string(N) + "_c" + std::to_string(C) 
          + "_h" + std::to_string(H) + "_w" + std::to_string(W) 
          + "_" + device_str;
          
  TVMHandlePtr instance = TVMHandle::getInstance();
  tvm::runtime::PackedFunc func = instance->get(func_name.c_str());
  if (func == nullptr) {
    // do search
    HICE_DLOG(INFO) << "searching " << func_name;  
    HICE_Py_Initialize();
    HicePyObject pModule = HICE_PyImport_ImportModule("relu_fwd_cuda");
    HicePyObject pFunc = HICE_PyObject_GetAttrString(pModule, "search_schedule");
    HicePyObject pArgsDict = HICE_PyDict_New();
    HICE_PyDict_SetItemString(pArgsDict, "N", HICE_Py_BuildValue("i", N));
    HICE_PyDict_SetItemString(pArgsDict, "C", HICE_Py_BuildValue("i", C));
    HICE_PyDict_SetItemString(pArgsDict, "H", HICE_Py_BuildValue("i", H));
    HICE_PyDict_SetItemString(pArgsDict, "W", HICE_Py_BuildValue("i", W));
    HicePyObject pArgsTuple = HICE_PyTuple_New(4);
    SetPargs(pArgsTuple, pArgsDict, func_name);
    HicePyObject pReturn = HICE_PyObject_CallObject(pFunc, pArgsTuple);
    HICE_Py_Finalize();
    HICE_DLOG(INFO) << "Python Finished.";
    // update
    instance->clear_module();
    func = instance->get(func_name.c_str());
    if (func == nullptr) {
      HICE_LOG(WARNING) << "Failed to generator tvm kernel: " << func_name;
      return false;
    }
  }

  DLManagedTensor dlm_input = HICETensor_to_DLManagedTensor(input);
  DLManagedTensor dlm_output = HICETensor_to_DLManagedTensor(output);
  tvm::runtime::NDArray in_tvm = runtime::NDArray::FromDLPack(&dlm_input);
  tvm::runtime::NDArray out_tvm = runtime::NDArray::FromDLPack(&dlm_output);
  func(in_tvm, out_tvm);
  if (input.device_type() == kCUDA) {
    TVMSynchronize(kDLGPU, 0, nullptr);
  }
  return true;
}


} // namespace hice