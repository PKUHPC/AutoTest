#ifdef HICE_USE_MKLDNN

#include "hice/device/cpu/context_cpu.h"
#include "hice/nn/softmax.h"

namespace hice {

namespace {

void softmax_fwd_impl(const Tensor &input, int64_t axis, Tensor &output) {
  // std::cout << "mkldnn fwd softmax"<< std::endl;
  int true_axis = input.get_true_axis(axis);
  int outer_size = input.size_to_dim(true_axis);
  int axis_size = input.dim(true_axis);
  int inner_size = input.size_from_dim(true_axis + 1);

  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  auto dims = dnnl::memory::dims{outer_size, axis_size, inner_size};
  switch (input.scalar_type()) {
    case kFloat: {
      using scalar_t = float;
      auto mem_desc = dnnl::memory::desc(
          dims, MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 3));
      auto src =
              dnnl::memory(mem_desc, engine, const_cast<Tensor &>(input).mutable_data<scalar_t>());
      auto dst =
              dnnl::memory(mem_desc, engine, output.mutable_data<scalar_t>());
      auto softmax_fwd_desc = dnnl::softmax_forward::desc(
              dnnl::prop_kind::forward_training, mem_desc, 1);
      auto softmax_fwd_pdesc =
              dnnl::softmax_forward::primitive_desc(softmax_fwd_desc,
                                                    engine);
      auto softmax_fwd = dnnl::softmax_forward(softmax_fwd_pdesc);
      softmax_fwd.execute(stream,
                          {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
      break;
    }
    default:
      HICE_LOG(ERROR) << "softmax does not support types other than float.";
      break;
  }
}

void softmax_bwd_impl(const Tensor &output, const Tensor &grad_output,
                      int64_t axis, Tensor &grad_input) {
  // std::cout << "mkldnn bwd softmax"<< std::endl;
  int true_axis = output.get_true_axis(axis);
  int outer_size = output.size_to_dim(true_axis);
  int axis_size = output.dim(true_axis);
  int inner_size = output.size_from_dim(true_axis + 1);

  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  auto dims = dnnl::memory::dims{outer_size, axis_size, inner_size};
  switch (output.scalar_type()) {
    case kFloat: {
      using scalar_t = float;
      auto mem_desc = dnnl::memory::desc(
          dims, MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 3));
      auto dst =
              dnnl::memory(mem_desc, engine,
                         const_cast<Tensor &>(output).mutable_data<scalar_t>());
      auto diff_dst = dnnl::memory(
          mem_desc, engine,
          const_cast<Tensor &>(grad_output).mutable_data<scalar_t>());
      auto diff_src =
              dnnl::memory(mem_desc, engine, grad_input.mutable_data<scalar_t>());
      auto softmax_fwd_desc = dnnl::softmax_forward::desc(
              dnnl::prop_kind::forward_training, mem_desc, 1);
      auto softmax_fwd_pdesc =
              dnnl::softmax_forward::primitive_desc(softmax_fwd_desc, engine);
      auto softmax_bwd_desc =
              dnnl::softmax_backward::desc(mem_desc, mem_desc, 1);
      auto softmax_bwd_pdesc = dnnl::softmax_backward::primitive_desc(
          softmax_bwd_desc, engine, softmax_fwd_pdesc);
      auto softmax_bwd = dnnl::softmax_backward(softmax_bwd_pdesc);
      softmax_bwd.execute(stream, {{DNNL_ARG_DST, dst},
                                   {DNNL_ARG_DIFF_DST, diff_dst},
                                   {DNNL_ARG_DIFF_SRC, diff_src}});
      break;
    }
    default:
      HICE_LOG(ERROR) << "softmax does not support types other than float.";
      break;
  }
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(softmax_fwd_dispatcher, &softmax_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense}   // output
);

HICE_REGISTER_KERNEL(softmax_bwd_dispatcher, &softmax_bwd_impl,
                     {kCPU, kDense},  // output
                     {kCPU, kDense},  // grad_output
                     {kCPU, kDense}   // grad_input
);

}  // namespace hice

#endif