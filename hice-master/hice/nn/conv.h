#pragma once 

#include "hice/util/types.h"
#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"

namespace hice {

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;


// NB: conv_output_dims and conv_input_dims are not bijections,
// as conv_output_dims loses information; this is why conv_input_dims
// takes an extra output_padding argument to resolve the ambiguity.

inline std::vector<int64_t> conv_output_dims(ConstIntArrayRef input_dims,
                                             ConstIntArrayRef weight_dims,
                                             ConstIntArrayRef padding,
                                             ConstIntArrayRef stride,
                                             ConstIntArrayRef dilation,
                                             int64_t groups) {
  auto dim = input_dims.size();
  std::vector<int64_t> output_dims(dim);
  output_dims[0] = input_dims[input_batch_size_dim];
  output_dims[1] = weight_dims[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_dims[d] - 1) + 1;
    output_dims[d] =
        (input_dims[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_dims;
}

inline std::vector<int64_t> conv_input_dims(
    ConstIntArrayRef output_dims, ConstIntArrayRef weight_dims,
    ConstIntArrayRef padding, ConstIntArrayRef output_padding,
    ConstIntArrayRef stride, ConstIntArrayRef dilation, int64_t groups) {
  auto dim = output_dims.size();
  std::vector<int64_t> input_dims(dim);
  input_dims[0] = output_dims[output_batch_size_dim];
  input_dims[1] = weight_dims[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_dims[d] - 1) + 1;
    input_dims[d] = (output_dims[d] - 1) * stride[d - 2] -
                    (2 * padding[d - 2]) + kernel + output_padding[d - 2];
  }
  return input_dims;
}

inline std::vector<int64_t> conv_weight_dims(
    ConstIntArrayRef input_dims, ConstIntArrayRef output_dims,
    ConstIntArrayRef padding, ConstIntArrayRef output_padding,
    ConstIntArrayRef stride, ConstIntArrayRef dilation, int64_t groups) {
  auto dim = input_dims.size();
  std::vector<int64_t> weight_dims(dim);
  weight_dims[0] = output_dims[1];
  weight_dims[1] = input_dims[1] / groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = input_dims[d] - (output_dims[d] - 1) * stride[d - 2] +
                 2 * padding[d - 2] - output_padding[d - 2];
    weight_dims[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_dims;
}

// Forward dispatcher
using conv_fwd_kernel_fn_type =
    void (*)(const Tensor &input, const Tensor &weight,
             hice::optional<Tensor> bias, ConstIntArrayRef padding,
             ConstIntArrayRef stride, ConstIntArrayRef dilation, int64_t groups,
             bool benchmark, bool deterministic, Tensor &output);

HICE_DECLARE_DISPATCHER(conv_fwd_dispatcher, conv_fwd_kernel_fn_type);

HICE_API Tensor conv_fwd(const Tensor &input, const Tensor &weight,
                         hice::optional<Tensor> bias, ConstIntArrayRef padding,
                         ConstIntArrayRef stride, ConstIntArrayRef dilation,
                         int64_t groups, bool benchmark, bool deterministic);

HICE_API Tensor &conv_fwd(const Tensor &input, const Tensor &weight,
                          hice::optional<Tensor> bias, ConstIntArrayRef padding,
                          ConstIntArrayRef stride, ConstIntArrayRef dilation,
                          int64_t groups, bool benchmark, bool deterministic,
                          Tensor &output);

// Backward dispatcher
using conv_bwd_kernel_fn_type = void (*)(
    const Tensor &input, const Tensor &weight, const Tensor &grad_output,
    ConstIntArrayRef padding, ConstIntArrayRef stride,
    ConstIntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, hice::optional<Tensor> grad_input,
    hice::optional<Tensor> grad_weight, hice::optional<Tensor> grad_bias);

HICE_DECLARE_DISPATCHER(conv_bwd_dispatcher, conv_bwd_kernel_fn_type);

HICE_API std::tuple<Tensor, Tensor, Tensor> conv_bwd(
    const Tensor &input, const Tensor &weight, const Tensor &grad_output,
    ConstIntArrayRef padding, ConstIntArrayRef stride,
    ConstIntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, std::array<bool, 3> output_mask);

HICE_API std::tuple<Tensor &, Tensor &, Tensor &> conv_bwd(
    const Tensor &input, const Tensor &weight, const Tensor &grad_output,
    ConstIntArrayRef padding, ConstIntArrayRef stride,
    ConstIntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic,
    hice::optional<Tensor> grad_input,
    hice::optional<Tensor> grad_weight,
    hice::optional<Tensor> grad_bias);

} // namespace hice
