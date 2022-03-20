#include <hice/basic/factories.h>
#include <hice/core/tensor_printer.h>

#include "hice/tvm/tvm.h"
#include <hice/nn/separable_conv.h>
#include <hice/math/unary_expr.h>
#include <hice/nn/conv.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

#if 0
// fname test
int main(int argc, char * argv[]) {
  using namespace hice;

  std::string device_str = 1 ? "cuda" : "cpu";

  std::cout << device_str << std::endl;

  return 0;
}
#endif


#if 0
// prelu test
int main(int argc, char * argv[]) {
  using namespace hice;

  tvm::runtime::Module module = tvm::runtime::Module::LoadFromFile("/home/amax101/hice/likesen/tvm_cooking/auto_scheduler/a.o.so", "so");
  tvm::runtime::Module module_device = tvm::runtime::Module::LoadFromFile("/home/amax101/hice/likesen/tvm_cooking/auto_scheduler/a.ptx");
  module->Import(module_device);
  tvm::runtime::PackedFunc pfunc = module->GetFunction("prelu");

  float in_data[] = {0.65561146,0.5935806,0.65862685,0.78939784,0.3373951,0.025202278,0.9808423,0.40788382,0.530461,0.046321232,0.48402992,0.8336426,0.3475852,0.30598977,0.40660256,0.67973703,0.56118846,0.57996505,0.09853239,0.5651121,0.880903,0.4896007,0.3034379,0.9452218,0.7555289,0.42474887,0.03529255,0.018169396,0.4146509,0.6936907,0.8494903,0.5414237};
  float alph_data[] = {0.11596452, 0.88517183};

  hice::Tensor input = hice::wrap({4, 2, 2, 2}, in_data, device(kCPU).dtype(kFloat)).to(kCUDA);
  hice::Tensor alphas = hice::wrap({2}, alph_data, device(kCPU).dtype(kFloat)).to(kCUDA);
  hice::Tensor output = hice::rand_uniform({4, 2, 2, 2}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  // hice::Tensor output = hice::empty({4, 2, 2, 2}, device(kCUDA).dtype(kFloat));

  runtime::NDArray A = hice::HICETensor_to_NDArray(input);
  runtime::NDArray B = hice::HICETensor_to_NDArray(alphas);
  runtime::NDArray C = hice::HICETensor_to_NDArray(output);
  
  pfunc(A, B, C);
  
  TensorPrinter tp;
  tp.print(output);
  return 0;
}
#endif

# if 0
// conv_fwd test
int main(int argc, char * argv[]) {
  using namespace hice;

  
  // tvm::runtime::Module module = tvm::runtime::Module::LoadFromFile("/home/amax101/hice/likesen/tvm_cooking/auto_scheduler/lib/test.so", "so");
  // tvm::runtime::Module module_device = tvm::runtime::Module::LoadFromFile("/home/amax101/hice/likesen/tvm_cooking/auto_scheduler/lib/conv_fwd_n32_c3_h5_w5_co2_k3_s1_p1_cuda.ptx");
  // module->Import(module_device);
  // tvm::runtime::PackedFunc pfunc = module->GetFunction("conv_fwd_n32_c3_h5_w5_co2_k3_s1_p1_cuda");

  int64_t CI = 3, CO = 32, HWI = 224, HWO = 224, ksize = 3, strid = 1, pad = 1;

  hice::Tensor input = hice::rand_uniform({32, CI, HWI, HWI}, -10.0, 10.0, device({kCUDA, 0}).dtype(kFloat));
  hice::Tensor kernel = hice::rand_uniform({CO, CI, ksize, ksize}, -10.0, 10.0, device({kCUDA, 0}).dtype(kFloat));
  hice::Tensor output = hice::rand_uniform({32, CO, HWO, HWO}, -10.0, 10.0, device({kCUDA, 0}).dtype(kFloat));

  int64_t stride[2] = {strid, strid};
  int64_t padding[2] = {pad, strid};
  int64_t dilation[2] = {1, 1};
  int64_t in_channel = input.dim(1);
  int64_t out_channel = output.dim(1);
  
  // CUDAContext ctx(0);
  // cudaSetDevice(0);
  // cudaStream_t strm(0);
  std::string device_str = input.device_type() == hice::kCUDA ? "cuda" : "cpu";
  std::string func_name = "conv_fwd_n" + std::to_string(32) + "_c" + std::to_string(CI) 
          + "_h" + std::to_string(HWI) + "_w" + std::to_string(HWI) 
          + "_co" + std::to_string(CO) + "_k" + std::to_string(ksize) 
          + "_s" + std::to_string(strid) + "_p" + std::to_string(pad) 
          + "_" + device_str;
  std::cout << func_name << std::endl;

  hice::conv_fwd(input, kernel, hice::nullopt, padding, stride, dilation, 1, false, true, output);
  // cudaStreamSynchronize(strm);
  // std::cout << __LINE__ << std::endl;

  // runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
  // runtime::NDArray krn_tvm = hice::HICETensor_to_NDArray(kernel);
  // runtime::NDArray out_tvm = hice::HICETensor_to_NDArray(output);
  // pfunc(in_tvm, krn_tvm, out_tvm);

  // TensorPrinter tp;
  // tp.print(output);

  return 0;

}
#endif 


# if 0
// sepconv test
int main(int argc, char * argv[]) {
  using namespace hice;

  // hice::Tensor input = hice::rand_uniform({32, 32, 112, 112}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  // hice::Tensor dkernel = hice::rand_uniform({32, 1, 3, 3}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  // hice::Tensor pkernel = hice::rand_uniform({64, 32, 1, 1}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  // hice::Tensor output = hice::empty({32, 64, 112, 112}, device(kCUDA).dtype(kFloat));

  // int64_t CI = 64, CO = 128, HWI = 112, HWO = 56, strd = 2;
  // int64_t CI = 128, CO = 128, HWI = 56, HWO = 56, strd = 1;
  // int64_t CI = 128, CO = 256, HWI = 56, HWO = 28, strd = 2;
  // int64_t CI = 256, CO = 256, HWI = 28, HWO = 28, strd = 1;
  // int64_t CI = 256, CO = 512, HWI = 28, HWO = 14, strd = 2;
  // int64_t CI = 512, CO = 512, HWI = 14, HWO = 14, strd = 1;
  // int64_t CI = 512, CO = 1024, HWI = 14, HWO = 7, strd = 2;
  int64_t CI = 1024, CO = 1024, HWI = 7, HWO = 7, strd = 1;

  hice::Tensor input = hice::rand_uniform({32, CI, HWI, HWI}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  hice::Tensor dkernel = hice::rand_uniform({CI, 1, 3, 3}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  hice::Tensor pkernel = hice::rand_uniform({CO, CI, 1, 1}, -10.0, 10.0, device(kCUDA).dtype(kFloat));
  hice::Tensor output = hice::empty({32, CO, HWO, HWO}, device(kCUDA).dtype(kFloat));

  int64_t stride[1] = {strd};
  int64_t padding[1] = {1};
  int64_t in_channel = input.dim(1);
  int64_t out_channel = output.dim(1);
  std::string func_name = std::string("sepconv_") + std::to_string(in_channel) + "_" + 
    std::to_string(out_channel) + "_" + std::to_string(padding[0]) + "_" + std::to_string(stride[0]);
  std::cout << func_name << std::endl;

  hice::separable_conv_fwd(input, dkernel, pkernel, padding, stride, output);
  // hice::exp(input);
  
  // TensorPrinter tp;
  // tp.print(output);

  return 0;
}
#endif 