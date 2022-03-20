//#if 1
 #ifdef HICE_USE_MKLDNN
#include "hice/nn/conv.h"
#include "hice/device/cpu/common_mkldnn.h"
#include "hice/device/cpu/context_cpu.h"
#include <algorithm>

namespace hice {
namespace {
using namespace dnnl;

void conv_fwd_impl(const Tensor& input, const Tensor& weight,
                   hice::optional<Tensor> bias, ConstIntArrayRef padding,
                   ConstIntArrayRef stride, ConstIntArrayRef dilation,
                   int64_t groups, bool benchmark, bool deterministic,
                   Tensor& output) {                     
  //std::cout << "cpu conv fwd" << std::endl;
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  std::vector<std::unordered_map<int, memory>> net_args;

  int64_t g = groups;
  int64_t sp_ndim = input.ndim() - 2;

  HICE_CHECK_EQ(input.dim(0), output.dim(0));
  HICE_CHECK_EQ(input.dim(1), weight.dim(1) * g);
  HICE_CHECK_EQ(output.dim(1), weight.dim(0));
  HICE_CHECK_EQ(sp_ndim, padding.size());
  HICE_CHECK_EQ(sp_ndim, stride.size());
  HICE_CHECK_EQ(sp_ndim, dilation.size());

  memory::dims input_dims(input.dims().begin(), input.dims().end());
  memory::dims weight_dims(weight.dims().begin(), weight.dims().end());
  if (g != 1) {
    weight_dims[0] /= g;
    weight_dims.insert(weight_dims.begin(), g);
  }
  memory::dims output_dims(output.dims().begin(), output.dims().end());
  memory::dims bias_dims{output.dim(1)};
  memory::dims padding_dims(padding.begin(), padding.end());
  memory::dims stride_dims(stride.begin(), stride.end());
  memory::dims dilation_dims(dilation.begin(), dilation.end());
  //std::transform(dilation_dims.begin(), dilation_dims.end(),
  //               dilation_dims.begin(), [](int x) -> int { return (x - 1); });
  for (auto &i : dilation_dims) i -= 1;

  auto data_t = get_mkldnn_data_type(input.scalar_type());
  auto format_any = memory::format_tag::any;
  auto format_nchw = get_mkldnn_format_tag(input.ndim());
  auto format_weight = get_mkldnn_format_tag(weight.ndim(), g);
  auto input_md = memory::desc({input_dims}, data_t, format_any);
  auto weight_md = memory::desc({weight_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto output_md = memory::desc({output_dims}, data_t, format_any);

  std::shared_ptr<convolution_forward::desc> conv_fwd_desc;
  if (bias) {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, bias_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  } else {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_fwd_pd;
  conv_fwd_pd.reset(
      new convolution_forward::primitive_desc(*conv_fwd_desc, engine));

  auto input_usr_memory = memory(
      {{input_dims}, data_t, format_nchw}, engine, const_cast<Tensor &>(input).raw_mutable_data());
  auto weight_usr_memory = memory(
      {{weight_dims}, data_t, format_weight}, engine, const_cast<Tensor &>(weight).raw_mutable_data());
  auto output_usr_memory = memory({{output_dims}, data_t, format_nchw}, engine,
                                  output.raw_mutable_data());

  std::vector<primitive> net;

  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != conv_fwd_pd->src_desc()) {
    input_memory = memory(conv_fwd_pd->src_desc(), engine);
    net.push_back(reorder(input_usr_memory, input_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, input_usr_memory}, {DNNL_ARG_TO, input_memory}});
  }

  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != conv_fwd_pd->weights_desc()) {
    weight_memory = memory(conv_fwd_pd->weights_desc(), engine);
    net.push_back(reorder(weight_usr_memory, weight_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto output_memory = output_usr_memory;
  if (output_memory.get_desc() != conv_fwd_pd->dst_desc()) {
    output_memory = memory(conv_fwd_pd->dst_desc(), engine);
  }

  std::shared_ptr<convolution_forward> conv_fwd;
  if (bias) {
    auto bias_memory = memory({{bias_dims}, data_t, memory::format_tag::x},
                              engine, bias.value().raw_mutable_data());
    conv_fwd.reset(new convolution_forward(*conv_fwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, input_memory},
                        {DNNL_ARG_WEIGHTS, weight_memory},
                        {DNNL_ARG_BIAS, bias_memory},
                        {DNNL_ARG_DST, output_memory}});
  } else {
    conv_fwd.reset(new convolution_forward(*conv_fwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, input_memory},
                        {DNNL_ARG_WEIGHTS, weight_memory},
                        {DNNL_ARG_DST, output_memory}});
  }
  net.push_back(*conv_fwd);

  if (output_memory != output_usr_memory) {
    net.push_back(reorder(output_memory, output_usr_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, output_memory}, {DNNL_ARG_TO, output_usr_memory}});
  }

  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(stream, net_args.at(i));
  stream.wait();
}

void conv_bwd_input(const Tensor& grad_output, const Tensor& weight,
                    hice::optional<Tensor> grad_bias, ConstIntArrayRef padding,
                    ConstIntArrayRef stride, ConstIntArrayRef dilation,
                    int64_t groups, Tensor& grad_input) {
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  std::vector<std::unordered_map<int, memory>> net_args;

  int64_t g = groups;
  int64_t sp_ndim = grad_input.ndim() - 2;

  HICE_CHECK_EQ(grad_input.dim(0), grad_output.dim(0));
  HICE_CHECK_EQ(grad_input.dim(1), weight.dim(1) * g);
  HICE_CHECK_EQ(grad_output.dim(1), weight.dim(0));
  HICE_CHECK_EQ(sp_ndim, padding.size());
  HICE_CHECK_EQ(sp_ndim, stride.size());
  HICE_CHECK_EQ(sp_ndim, dilation.size());

  memory::dims input_dims(grad_input.dims().begin(), grad_input.dims().end());
  memory::dims weight_dims(weight.dims().begin(),
                           weight.dims().end());
  if (g != 1) {
    weight_dims[0] /= g;
    weight_dims.insert(weight_dims.begin(), g);
  }
  memory::dims output_dims(grad_output.dims().begin(),
                           grad_output.dims().end());
  ////std::cout << "grad_output" << std::endl;
  //for (auto i : output_dims) {
  //  //std::cout << i << ", ";
  //}
  ////std::cout << std::endl;
  memory::dims bias_dims{grad_output.dim(1)};
  memory::dims padding_dims(padding.begin(), padding.end());
  memory::dims stride_dims(stride.begin(), stride.end());
  memory::dims dilation_dims(dilation.begin(), dilation.end());
  for (auto &i : dilation_dims) i -= 1;

  auto data_t = get_mkldnn_data_type(grad_input.scalar_type());
  auto format_nchw = get_mkldnn_format_tag(grad_input.ndim());
  auto format_weight = get_mkldnn_format_tag(weight.ndim(), g);
  auto format_any = memory::format_tag::any;
  auto input_md = memory::desc({input_dims}, data_t, format_any);
  auto weight_md = memory::desc({weight_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto output_md = memory::desc({output_dims}, data_t, format_any);

  // need to re-create conv_fwd_pd to feed conv_bwd_data_pd
  std::shared_ptr<convolution_forward::desc> conv_fwd_desc;
  if (grad_bias) {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, bias_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  } else {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_fwd_pd;
  conv_fwd_pd.reset(
      new convolution_forward::primitive_desc(*conv_fwd_desc, engine));

  std::shared_ptr<convolution_backward_data::desc> conv_bwd_data_desc;
  conv_bwd_data_desc.reset(new convolution_backward_data::desc(
      algorithm::convolution_direct, input_md, weight_md, output_md,
      stride_dims, dilation_dims, padding_dims, padding_dims));

  std::shared_ptr<convolution_backward_data::primitive_desc> conv_bwd_data_pd;
  conv_bwd_data_pd.reset(new convolution_backward_data::primitive_desc(
      *conv_bwd_data_desc, engine, *conv_fwd_pd));

  auto grad_output_usr_memory = memory(
      {{output_dims}, data_t, format_nchw}, engine, const_cast<Tensor &>(grad_output).raw_mutable_data());
  auto weight_usr_memory =
      memory({{weight_dims}, data_t, format_weight}, engine, const_cast<Tensor &>(weight).raw_mutable_data());
  auto grad_input_usr_memory = memory({{input_dims}, data_t, format_nchw},
                                      engine, grad_input.raw_mutable_data());

  std::vector<primitive> net;

  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != conv_bwd_data_pd->diff_dst_desc()) {
    grad_output_memory = memory(conv_bwd_data_pd->diff_dst_desc(), engine);
    net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
    net_args.push_back({{DNNL_ARG_FROM, grad_output_usr_memory},
                        {DNNL_ARG_TO, grad_output_memory}});
  }

  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != conv_bwd_data_pd->weights_desc()) {
    weight_memory = memory(conv_bwd_data_pd->weights_desc(), engine);
    net.push_back(reorder(weight_usr_memory, weight_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != conv_bwd_data_pd->diff_src_desc()) {
    grad_input_memory = memory(conv_bwd_data_pd->diff_src_desc(), engine);
  }

  std::shared_ptr<convolution_backward_data> conv_bwd_data;
  conv_bwd_data.reset(new convolution_backward_data(
      *conv_bwd_data_pd));
  net.push_back(*conv_bwd_data);
  net_args.push_back({{DNNL_ARG_DIFF_SRC, grad_input_memory},
                      {DNNL_ARG_WEIGHTS, weight_memory},
                      {DNNL_ARG_DIFF_DST, grad_output_memory}});

  if (grad_input_memory != grad_input_usr_memory) {
    net.push_back(reorder(grad_input_memory, grad_input_usr_memory));
    net_args.push_back({{DNNL_ARG_FROM, grad_input_memory},
                        {DNNL_ARG_TO, grad_input_usr_memory}});
  }

  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(stream, net_args.at(i));
  stream.wait();
}

void conv_bwd_weights(const Tensor& grad_output, const Tensor& input,
                      ConstIntArrayRef padding, ConstIntArrayRef stride,
                      ConstIntArrayRef dilation, int64_t groups,
                      Tensor& grad_weight, hice::optional<Tensor> grad_bias) {
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  std::vector<std::unordered_map<int, memory>> net_args;

  int64_t g = groups;
  int64_t sp_ndim = input.ndim() - 2;

  HICE_CHECK_EQ(input.dim(0), grad_output.dim(0));
  HICE_CHECK_EQ(input.dim(1), grad_weight.dim(1) * g);
  HICE_CHECK_EQ(grad_output.dim(1), grad_weight.dim(0));
  HICE_CHECK_EQ(sp_ndim, padding.size());
  HICE_CHECK_EQ(sp_ndim, stride.size());
  HICE_CHECK_EQ(sp_ndim, dilation.size());

  memory::dims input_dims(input.dims().begin(), input.dims().end());
  memory::dims weight_dims(grad_weight.dims().begin(),
                           grad_weight.dims().end());
  if (g != 1) {
    weight_dims[0] /= g;
    weight_dims.insert(weight_dims.begin(), g);
  }
  memory::dims output_dims(grad_output.dims().begin(),
                           grad_output.dims().end());
  memory::dims bias_dims{grad_output.dim(1)};
  memory::dims padding_dims(padding.begin(), padding.end());
  memory::dims stride_dims(stride.begin(), stride.end());
  memory::dims dilation_dims(dilation.begin(), dilation.end());
  for (auto &i : dilation_dims) i -= 1;

  ////std::cout << "cpu conv bwd 1" << std::endl;
  auto data_t = get_mkldnn_data_type(input.scalar_type());
  auto format_nchw = get_mkldnn_format_tag(input.ndim());
  auto format_weight = get_mkldnn_format_tag(grad_weight.ndim(), g);
  auto format_any = memory::format_tag::any;
  auto input_md = memory::desc({input_dims}, data_t, format_any);
  auto weight_md = memory::desc({weight_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto output_md = memory::desc({output_dims}, data_t, format_any);

  // need to re-create conv_fwd_pd to feed conv_bwd_weight_pd
  std::shared_ptr<convolution_forward::desc> conv_fwd_desc;
  if (grad_bias) {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, bias_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  } else {
    conv_fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, input_md,
        weight_md, output_md, stride_dims, dilation_dims, padding_dims,
        padding_dims));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_fwd_pd;
  conv_fwd_pd.reset(
      new convolution_forward::primitive_desc(*conv_fwd_desc, engine));

  std::shared_ptr<convolution_backward_weights::desc> conv_bwd_weight_desc;
  if (grad_bias) {
    conv_bwd_weight_desc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct, input_md, weight_md, bias_md, output_md,
        stride_dims, dilation_dims, padding_dims, padding_dims));
  } else {
    conv_bwd_weight_desc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct, input_md, weight_md, output_md,
        stride_dims, dilation_dims, padding_dims, padding_dims));
  }

  std::shared_ptr<convolution_backward_weights::primitive_desc>
      conv_bwd_weight_pd;
  conv_bwd_weight_pd.reset(new convolution_backward_weights::primitive_desc(
      *conv_bwd_weight_desc, engine, *conv_fwd_pd));

  auto input_usr_memory =
      memory({{input_dims}, data_t, format_nchw}, engine, const_cast<Tensor &>(input).raw_mutable_data());
  auto grad_output_usr_memory = memory(
      {{output_dims}, data_t, format_nchw}, engine, const_cast<Tensor &>(grad_output).raw_mutable_data());
  auto grad_weight_usr_memory = memory({{weight_dims}, data_t, format_weight},
                                       engine, const_cast<Tensor &>(grad_weight).raw_mutable_data());

  std::vector<primitive> net;

  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != conv_bwd_weight_pd->src_desc()) {
    input_memory = memory(conv_bwd_weight_pd->src_desc(), engine);
    net.push_back(reorder(input_usr_memory, input_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, input_usr_memory}, {DNNL_ARG_TO, input_memory}});
  }

  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() !=
      conv_bwd_weight_pd->diff_dst_desc()) {
    grad_output_memory = memory(conv_bwd_weight_pd->diff_dst_desc(), engine);
    net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
    net_args.push_back({{DNNL_ARG_FROM, grad_output_usr_memory},
                        {DNNL_ARG_TO, grad_output_memory}});
  }

  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_desc() !=
      conv_bwd_weight_pd->diff_weights_desc()) {
    grad_weight_memory =
        memory(conv_bwd_weight_pd->diff_weights_desc(), engine);
  }

  std::shared_ptr<convolution_backward_weights> conv_bwd_weight;
  if (grad_bias) {
    auto grad_bias_memory =
        memory({{bias_dims}, data_t, memory::format_tag::x}, engine,
               grad_bias.value().raw_mutable_data());
    conv_bwd_weight.reset(
        new convolution_backward_weights(*conv_bwd_weight_pd));
    net_args.push_back({{DNNL_ARG_SRC, input_memory},
                        {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory},
                        {DNNL_ARG_DIFF_BIAS, grad_bias_memory},
                        {DNNL_ARG_DIFF_DST, grad_output_memory}});
    } else {
      conv_bwd_weight.reset(
          new convolution_backward_weights(*conv_bwd_weight_pd));
      net_args.push_back({{DNNL_ARG_SRC, input_memory},
                          {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory},
                          {DNNL_ARG_DIFF_DST, grad_output_memory}});
  }

  net.push_back(*conv_bwd_weight);

  if (grad_weight_memory != grad_weight_usr_memory) {
    net.push_back(reorder(grad_weight_memory, grad_weight_usr_memory));
    net_args.push_back({{DNNL_ARG_FROM, grad_weight_memory},
                        {DNNL_ARG_TO, grad_weight_usr_memory}});
  }

  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(stream, net_args.at(i));
  stream.wait();
}

void conv_bwd_impl(const Tensor& input, const Tensor& weight,
                   const Tensor& grad_output, ConstIntArrayRef padding,
                   ConstIntArrayRef stride, ConstIntArrayRef dilation,
                   int64_t groups, bool benchmark, bool deterministic,
                   hice::optional<Tensor> grad_input,
                   hice::optional<Tensor> grad_weight,
                   hice::optional<Tensor> grad_bias) {
  // std::cout << "cpu conv bwd" << std::endl;
  if (grad_input) {
     conv_bwd_input(grad_output, weight, grad_bias, padding, stride, dilation,
                   groups, grad_input.value());
  }
  if (grad_weight) {
    conv_bwd_weights(grad_output, input, padding, stride, dilation, groups,
                     grad_weight.value(), grad_bias);
  }
}

} // anonymous namespace

HICE_REGISTER_KERNEL(conv_fwd_dispatcher, 
                     &conv_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // weight
                     {kCPU, kDense}   // output
);

HICE_REGISTER_KERNEL(conv_bwd_dispatcher, 
                     &conv_bwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // weight
                     {kCPU, kDense}   // grad_output
);

} // namespace hice

#endif