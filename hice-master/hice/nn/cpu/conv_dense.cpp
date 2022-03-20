 #ifdef HICE_USE_CPU_NATIVE
//#if 1

#include <algorithm>
#include "hice/core/dimension.h"
#include "hice/core/index_util.h"
#include "hice/device/cpu/context_cpu.h"
#include "hice/nn/conv.h"

namespace hice {

namespace {

// infer the expected param from
inline static std::vector<int64_t> infer_params(ConstIntArrayRef origin_param,
                                                int64_t ndim_op,
                                                int64_t default_value) {
  auto length = origin_param.size();
  if (length == ndim_op) {
    return std::vector<int64_t>(origin_param.begin(), origin_param.end());
  } else if (length == 0) {
    return std::vector<int64_t>(ndim_op, default_value);
  } else if (length == 1) {
    return std::vector<int64_t>(ndim_op, origin_param[0]);
  } else {
    HICE_CHECK_ARGUMENT(false)
        << "Unexpected length of stride/padding/dilation array.";
  }
}

template <typename scalar_t>
void conv_fwd_dense(const Tensor& input, const Tensor& kernel,
                    const scalar_t* bias, const int64_t* padding,
                    const int64_t* stride, const int64_t* dilation,
                    int64_t groups, Tensor& output) {
  int64_t n_sample = input.dim(0);
  int64_t in_channels = input.dim(1);
  int64_t out_channels = kernel.dim(0);
  int64_t out_channel_size = hice::size_from_dim(2, output.dims());
  int64_t kernel_size = hice::size_from_dim(1, kernel.dims());
  int64_t ndim_conv = input.ndim() - 2;

  std::vector<int64_t> in_coord(input.ndim(), 0);
  std::vector<int64_t> out_coord(output.ndim(), 0);
  std::vector<int64_t> kernel_coord(kernel.ndim());

  const scalar_t* in_ptr = input.data<scalar_t>();
  const scalar_t* kernel_ptr = kernel.data<scalar_t>();
  scalar_t* out_ptr = output.mutable_data<scalar_t>();

  // for each element of output
  for (int64_t out_idx = 0; out_idx < output.size(); ++out_idx) {
    // std::cout<<"out_coord="<<out_coord<<std::endl;
    // get input's multi_index(part)
    in_coord[0] = out_coord[0];

    // get kernel's multi_index
    std::fill(kernel_coord.begin() + 1, kernel_coord.end(), 0);
    kernel_coord[0] = out_coord[1];

    // calculate
    scalar_t tmp = 0;
    scalar_t b_ = bias == nullptr ? 0 : bias[out_coord[1]];
    for (int64_t k_idx = 0; k_idx < kernel_size; ++k_idx) {
      // get input's multi_index and check out_of_bound
      in_coord[1] = (kernel_coord[0] / (out_channels / groups)) * (in_channels / groups) + kernel_coord[1];
      bool out_of_bound = false;
      for (int d = 0; d < ndim_conv; ++d) {
        in_coord[d + 2] = out_coord[d + 2] * stride[d] - padding[d] + kernel_coord[d + 2] * dilation[d];
        if (in_coord[d + 2] < 0 ||
            in_coord[d + 2] >= input.dim(d + 2)) {
          out_of_bound = true;
          break;
        }
      }

      // std::cout<<"kernel_coord="<<kernel_coord<<std::endl;
      // std::cout<<"in_coord="<<in_coord<<std::endl;
      if (!out_of_bound) {
        int64_t in_offset =
            IndexUtil::multi_index_to_offset(input.shape(), in_coord);
        int64_t kernel_offset =
            IndexUtil::multi_index_to_offset(kernel.shape(), kernel_coord);
        tmp += in_ptr[in_offset] * kernel_ptr[kernel_offset];
      }

      IndexUtil::next_multi_index(kernel.shape(), kernel_coord);
    }
    int64_t out_offset =
        IndexUtil::multi_index_to_offset(output.shape(), out_coord);
    out_ptr[out_offset] = tmp + b_;
    // next output
    IndexUtil::next_multi_index(output.shape(), out_coord);
  }
}

/** NOTE: parameters benchmark and deterministic
 *  are disabled in this native version.
 *
 */
void conv_fwd_impl(const Tensor& input, const Tensor& weight,
                   hice::optional<Tensor> bias, ConstIntArrayRef padding,
                   ConstIntArrayRef stride, ConstIntArrayRef dilation,
                   int64_t groups, bool benchmark, bool deterministic,
                   Tensor& output) {
  int64_t in_channels = input.dim(1);
  int64_t out_channels = weight.dim(0);
  const void* bias_ptr = nullptr;
  const int64_t* padding_ptr = padding.data();
  const int64_t* stride_ptr = stride.data();
  const int64_t* dilation_ptr = dilation.data();
  HICE_CHECK_DIMS_MATCH(in_channels == weight.dim(1) * groups)
      << "in_channels(=" << in_channels
      << ") is inconsistency between input and weight";
  HICE_CHECK_DIMS_MATCH(out_channels % groups == 0)
      << "out_channels(=" << out_channels
      << ") can not be divisible by groups(=" << groups
      << ")";
  HICE_CHECK_TYPE_MATCH(input.scalar_type() == weight.scalar_type());
  if (bias.has_value()) {
    Tensor bais_t = bias.value();
    HICE_CHECK_TYPE_MATCH(input.scalar_type() == bais_t.scalar_type());
    HICE_CHECK_DIMS_MATCH(out_channels == bais_t.size());
    bias_ptr = bais_t.raw_data();
  }

  HICE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_fwd_native", [&]() {
    conv_fwd_dense<scalar_t>(
        input, weight, reinterpret_cast<const scalar_t*>(bias_ptr), padding_ptr,
        stride_ptr, dilation_ptr, groups, output);
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(conv_fwd_dispatcher, &conv_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // weight
                     {kCPU, kDense}   // output
);

}  // namespace hice

#endif