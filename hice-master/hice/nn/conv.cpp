#include "hice/nn/conv.h"

#ifdef HICE_USE_TVM
#include <hice/tvm/tvm.h>
#include <sys/time.h>
// #define USE_TVM
#endif // HICE_USE_TVM

namespace hice {

// Forward dispatcher
HICE_DEFINE_DISPATCHER(conv_fwd_dispatcher);

Tensor conv_fwd(const Tensor &input, const Tensor &weight,
                         hice::optional<Tensor> bias, ConstIntArrayRef padding,
                         ConstIntArrayRef stride, ConstIntArrayRef dilation,
                         int64_t groups, bool benchmark, bool deterministic) {
  std::vector<int64_t> output_dims = conv_output_dims(
      input.dims(), weight.dims(), padding, stride, dilation, groups);
  //std::cout << std::endl;
  //std::cout << "dilation_dims" << std::endl;
  //for (auto i : dilation) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;
  Tensor output(output_dims, device(input.device()).dtype(input.data_type()));
  conv_fwd_dispatcher(input, weight, bias, padding, stride, dilation, groups,
                      benchmark, deterministic, output);
  return output;
}

Tensor &conv_fwd(const Tensor &input, const Tensor &weight,
                hice::optional<Tensor> bias, ConstIntArrayRef padding,
                ConstIntArrayRef stride, ConstIntArrayRef dilation,
                int64_t groups, bool benchmark, bool deterministic,
                Tensor &output) {
#ifdef USE_TVM

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
          
  tvm::runtime::PackedFunc func = TVMHandle::get(func_name.c_str());
  if (func != nullptr) {
    tvm::runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
    tvm::runtime::NDArray krn_tvm = hice::HICETensor_to_NDArray(weight);
    tvm::runtime::NDArray out_tvm = hice::HICETensor_to_NDArray(output);
    // timeval t1, t2;
    // gettimeofday(&t1, NULL);
    func(in_tvm, krn_tvm, out_tvm);
    // gettimeofday(&t2, NULL);
    // std::cout << "func:" << (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 << std::endl;

    if (input.device_type() == kCUDA) {
      TVMSynchronize(kDLGPU, 0, nullptr);
    }
    return output;
  }
  HICE_DLOG(INFO) << "Not find TVM kernel:" << func_name 
    << ", Fall back to hice_conv.";
#endif  // USE_TVM
  std::vector<int64_t> output_dims = conv_output_dims(
      input.dims(), weight.dims(), padding, stride, dilation, groups);
  HICE_CHECK_EQ(compare_dims(output_dims, output.dims()), 0);
  conv_fwd_dispatcher(input, weight, bias, padding, stride, dilation, groups,
                      benchmark, deterministic, output);
  return output;
}

// Backward dispatcher
HICE_DEFINE_DISPATCHER(conv_bwd_dispatcher);

std::tuple<Tensor, Tensor, Tensor> conv_bwd(
    const Tensor &input, const Tensor &weight, const Tensor &grad_output,
    ConstIntArrayRef padding, ConstIntArrayRef stride,
    ConstIntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, std::array<bool, 3> output_mask) {
  hice::optional<Tensor> grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input =
        Tensor(input.dims(), device(input.device()).dtype(input.data_type()));
  }
  if (output_mask[1]) {
    grad_weight = Tensor(weight.dims(),
                         device(weight.device()).dtype(weight.data_type()));
  }
  if (output_mask[2]) {
    grad_bias =
        Tensor({grad_output.dim(output_channels_dim)},
               device(grad_output.device()).dtype(grad_output.data_type()));
  }
  //std::cout << "grad_output" << std::endl;
  //for (auto i : grad_output.dims()) {
  //  std::cout << i << ", ";
  //}
  //std::cout << std::endl;
  conv_bwd_dispatcher(input, weight, grad_output, padding, stride, dilation,
                      groups, benchmark, deterministic, grad_input, grad_weight,
                      grad_bias);
  if (!output_mask[0]) {
    grad_input = Tensor();
  }
  if (!output_mask[1]) {
    grad_weight = Tensor();
  }
  if (!output_mask[2]) {
    grad_bias = Tensor();
  }
  return std::tuple<Tensor, Tensor, Tensor>{
      grad_input.value(), grad_weight.value(), grad_bias.value()};
}

std::tuple<Tensor &, Tensor &, Tensor &> conv_bwd(
    const Tensor &input, const Tensor &weight, const Tensor &grad_output,
    ConstIntArrayRef padding, ConstIntArrayRef stride,
    ConstIntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic,
    hice::optional<Tensor> grad_input,
    hice::optional<Tensor> grad_weight,
    hice::optional<Tensor> grad_bias) {
  // std::cout << "conv_bwd before dispath" << std::endl;

  conv_bwd_dispatcher(input, weight, grad_output, padding, stride, dilation,
                      groups, benchmark, deterministic, grad_input, grad_weight,
                      grad_bias);
  if (!grad_input) {
    grad_input = Tensor();
  }
  if (!grad_weight) {
    grad_weight = Tensor();
  }
  if (!grad_bias) {
    grad_bias = Tensor();
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{grad_input.value(), grad_weight.value(), grad_bias.value()};
}


} // namespace hice