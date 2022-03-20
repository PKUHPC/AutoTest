#include "hice/api_c/ops_nn.h"
#include "hice/api_c/error_handle.h"
#include "hice/api_c/tensor_impl.h"

#include "hice/nn/activation.h"
#include "hice/nn/conv.h"

HI_Status HI_Abs(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::abs_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Relu(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::relu_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Sigmoid(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::sigmoid_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Sqrt(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::sqrt_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Square(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::square_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Tanh(const HI_Tensor input, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::tanh_fwd(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Elu(const HI_Tensor input, const float alpha, HI_Tensor* output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::elu_fwd(input->tensor_, alpha)};
  HI_API_END();
}

HI_Status HI_Conv(const HI_Tensor input, const HI_Tensor kernel,
                  const int* stride, const int stride_len, const int* padding,
                  const int padding_len, const int* dilation,
                  const int dilation_len, const int group_count,
                  HI_Tensor* output) {
  HI_API_BEGIN();
  std::vector<int64_t> stride64(stride, stride + stride_len);
  std::vector<int64_t> padding64(padding, padding + padding_len);
  std::vector<int64_t> dilation64(dilation, dilation + dilation_len);
  *output = new HI_Tensor_Impl{
      hice::conv_fwd(input->tensor_, kernel->tensor_, hice::nullopt, padding64,
                     stride64, dilation64, group_count, false, false)};
  HI_API_END();
}