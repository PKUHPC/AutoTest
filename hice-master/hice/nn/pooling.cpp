#include "hice/nn/pooling.h"
#include "hice/basic/reshape.h"

namespace hice {

// AVG Forward
HICE_DEFINE_DISPATCHER(pooling_avg_fwd_dispatcher);

Tensor pooling_avg_fwd(const Tensor& input, ConstIntArrayRef kernel,
                       ConstIntArrayRef stride, ConstIntArrayRef padding) {
  // check params
  int dim_pooling = input.ndim() - 2;
  // kernel != null
  HICE_CHECK(kernel.size() == dim_pooling || kernel.size() == 1);
  auto kernel_inferred = infer_params(kernel, dim_pooling, DEFAULT_KERNEL_SIZE);
  auto stride_inferred = infer_params(stride, dim_pooling, DEFAULT_STRIDE);
  auto padding_inferred = infer_params(padding, dim_pooling, DEFAULT_PADDING);
  auto output_dims = compute_out_dims(input.dims(), kernel_inferred, padding_inferred, stride_inferred);
  Tensor output(output_dims, device(input.device()).dtype(input.data_type()).layout(kDense));
  // convert 1d to 2d
  if (dim_pooling == 1) {
    kernel_inferred.push_back(DEFAULT_KERNEL_SIZE);
    stride_inferred.push_back(DEFAULT_STRIDE);
    padding_inferred.push_back(DEFAULT_PADDING);
    Tensor input_new = expand_dims(input, -1);
    Tensor output_new = expand_dims(output, -1);
    pooling_avg_fwd_dispatcher(input_new, kernel_inferred, stride_inferred, padding_inferred, output_new);
  } else {
    pooling_avg_fwd_dispatcher(input, kernel_inferred, stride_inferred, padding_inferred, output);
  }
  return output;
}

Tensor& pooling_avg_fwd(const Tensor &input, ConstIntArrayRef kernel,
                        ConstIntArrayRef stride, ConstIntArrayRef padding,
                        Tensor &output) {
  // check params
  int dim_pooling = input.ndim() - 2;
  // kernel != null
  HICE_CHECK(kernel.size() == dim_pooling || kernel.size() == 1);
  auto kernel_inferred = infer_params(kernel, dim_pooling, DEFAULT_KERNEL_SIZE);
  auto stride_inferred = infer_params(stride, dim_pooling, DEFAULT_STRIDE);
  auto padding_inferred = infer_params(padding, dim_pooling, DEFAULT_PADDING);
  auto output_dims = compute_out_dims(input.dims(), kernel_inferred, padding_inferred, stride_inferred);
  HICE_CHECK_EQ(compare_dims(output_dims, output.dims()), 0);
  // convert 1d to 2d
  if (dim_pooling == 1) {
    kernel_inferred.push_back(DEFAULT_KERNEL_SIZE);
    stride_inferred.push_back(DEFAULT_STRIDE);
    padding_inferred.push_back(DEFAULT_PADDING);
    Tensor input_new = expand_dims(input, -1);
    Tensor output_new = expand_dims(output, -1);
    pooling_avg_fwd_dispatcher(input_new, kernel_inferred, stride_inferred, padding_inferred, output_new);
  } else {
    pooling_avg_fwd_dispatcher(input, kernel_inferred, stride_inferred, padding_inferred, output);
  }
  return output;
}



// AVG Backward
HICE_DEFINE_DISPATCHER(pooling_avg_bwd_dispatcher);

Tensor pooling_avg_bwd(const Tensor& input, const Tensor& output,
                       const Tensor& grad_output, ConstIntArrayRef kernel,
                       ConstIntArrayRef stride, ConstIntArrayRef padding) {
  // check params
  int dim_pooling = input.ndim() - 2;
  // kernel != null
  HICE_CHECK(kernel.size() == dim_pooling || kernel.size() == 1);
  auto kernel_inferred = infer_params(kernel, dim_pooling, DEFAULT_KERNEL_SIZE);
  auto stride_inferred = infer_params(stride, dim_pooling, DEFAULT_STRIDE);
  auto padding_inferred = infer_params(padding, dim_pooling, DEFAULT_PADDING);
  Tensor grad_input(
      input.dims(), device(input.device()).dtype(input.data_type()).layout(kDense));
  if (dim_pooling == 1) {
    kernel_inferred.push_back(DEFAULT_KERNEL_SIZE);
    stride_inferred.push_back(DEFAULT_STRIDE);
    padding_inferred.push_back(DEFAULT_PADDING);
    Tensor input_new = expand_dims(input, -1);
    Tensor output_new = expand_dims(output, -1);
    Tensor grad_output_new = expand_dims(grad_output, -1);
    Tensor grad_input_new = expand_dims(grad_input, -1);
    pooling_avg_bwd_dispatcher(input_new, output_new, grad_output_new, kernel_inferred, stride_inferred,
                             padding_inferred, grad_input_new);
  } else {
    pooling_avg_bwd_dispatcher(input, output, grad_output, kernel_inferred, stride_inferred,
                              padding_inferred, grad_input);
  }
  return grad_input;
}

Tensor& pooling_avg_bwd(const Tensor& input, const Tensor& output,
                        const Tensor& grad_output,
                        ConstIntArrayRef kernel, ConstIntArrayRef stride,
                        ConstIntArrayRef padding, Tensor& grad_input) {
#ifdef USE_TVM

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

  tvm::runtime::PackedFunc func = TVMHandle::get(func_name.c_str());
  if (func != nullptr) {
    tvm::runtime::NDArray in_tvm = hice::HICETensor_to_NDArray(input);
    tvm::runtime::NDArray grad_out_tvm = hice::HICETensor_to_NDArray(grad_output);
    tvm::runtime::NDArray grad_in_tvm = hice::HICETensor_to_NDArray(grad_input);
    func(in_tvm, grad_out_tvm, grad_in_tvm); 
    if (input.device_type() == kCUDA) {
      TVMSynchronize(kDLGPU, 0, nullptr);
    }
    return grad_input;
  }

  HICE_DLOG(INFO) << "Not find TVM kernel:" << func_name 
    << ", Fall back to hice operator.";
#endif  // USE_TVM
  // check params
  int dim_pooling = input.ndim() - 2;
  HICE_CHECK_EQ(compare_dims(input.dims(), grad_input.dims()), 0);
  // kernel != null
  HICE_CHECK(kernel.size() == dim_pooling || kernel.size() == 1);
  auto kernel_inferred = infer_params(kernel, dim_pooling, DEFAULT_KERNEL_SIZE);
  auto stride_inferred = infer_params(stride, dim_pooling, DEFAULT_STRIDE);
  auto padding_inferred = infer_params(padding, dim_pooling, DEFAULT_PADDING);
  if (dim_pooling == 1) {
    kernel_inferred.push_back(DEFAULT_KERNEL_SIZE);
    stride_inferred.push_back(DEFAULT_STRIDE);
    padding_inferred.push_back(DEFAULT_PADDING);
    Tensor input_new = expand_dims(input, -1);
    Tensor output_new = expand_dims(output, -1);
    Tensor grad_output_new = expand_dims(grad_output, -1);
    Tensor grad_input_new = expand_dims(grad_input, -1);
    pooling_avg_bwd_dispatcher(input_new, output_new, grad_output_new, kernel_inferred, stride_inferred,
                             padding_inferred, grad_input_new);
  } else {
    pooling_avg_bwd_dispatcher(input, output, grad_output, kernel, stride,
                              padding, grad_input);
  }
  return grad_input;
}



// MAX Forward
HICE_DEFINE_DISPATCHER(pooling_max_fwd_dispatcher);

std::tuple<Tensor, Tensor> pooling_max_fwd(const Tensor& input,
                                           ConstIntArrayRef kernel,
                                           ConstIntArrayRef stride,
                                           ConstIntArrayRef padding) {
  Tensor output(device(input.device()).dtype(input.data_type()).layout(kDense));
  Tensor indices(device(input.device()).dtype(kInt32).layout(kDense));
  pooling_max_fwd_dispatcher(input, kernel, stride, padding, indices, output,
                             /* resizable = */ true);
  return std::make_tuple(output, indices);
}



// MAX Backward
HICE_DEFINE_DISPATCHER(pooling_max_bwd_dispatcher);

Tensor pooling_max_bwd(const Tensor& input, const Tensor& output,
                       const Tensor& grad_output, const Tensor& indices,
                       ConstIntArrayRef kernel, ConstIntArrayRef stride,
                       ConstIntArrayRef padding) {
  Tensor grad_input(
      device(input.device()).dtype(input.data_type()).layout(kDense));
  pooling_max_bwd_dispatcher(input, output, grad_output, indices, kernel,
                             stride, padding, grad_input,
                             /* resizable = */ true);
  return grad_input;
}

}  // namespace hice
