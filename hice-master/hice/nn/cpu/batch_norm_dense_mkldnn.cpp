#ifdef HICE_USE_MKLDNN

#include <vector>

#include "hice/device/cpu/context_cpu.h"
#include "hice/basic/factories.h"
#include "hice/nn/batch_norm.h"

/*
using mkldnn::batch_normalization_backward;
using mkldnn::batch_normalization_forward;
using mkldnn::prop_kind;
using mkldnn::stream;
using mkldnn::use_global_stats;
using mkldnn::use_scale_shift;
*/
namespace hice {

namespace {

void batch_norm_fwd_impl(Tensor& input, Tensor& output, Tensor& bn_scale,
                         Tensor& bn_bias, Tensor& running_mean,
                         Tensor& running_var, bool train, int mode,
                         double epsilon, double expo_factor, Tensor& saved_mean,
                         Tensor& saved_inv_var) {
  // sanity check
  auto input_shape = input.dims();

  /* to be finished*/
  // setup
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  using scalar_t = float;

    dnnl::memory::desc* mem_desc;
  if (input_shape.size() == 4) {
    if (mode == HICE_BATCHNORM_PER_ACTIVATION) {
      //          dims =
      //          mkldnn::memory::dims{input_shape[0],input_shape[1]*input_shape[2]*input_shape[3]};
      mem_desc = new dnnl::memory::desc(
          {input_shape[0], input_shape[1] * input_shape[2] * input_shape[3]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 2));
    } else if (mode == HICE_BATCHNORM_SPATIAL) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0], input_shape[1], input_shape[2], input_shape[3]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 4));
    } else {  // mode error
    }
  } else if (input_shape.size() == 5) {
    if (mode == HICE_BATCHNORM_PER_ACTIVATION) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0],
           input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 2));
    } else if (mode == HICE_BATCHNORM_SPATIAL) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0],
           input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 5));
    } else {  // mode error
    }
  } else {  // dim error
  }

  auto src = dnnl::memory(*mem_desc, engine, input.mutable_data<scalar_t>());
  auto dst = dnnl::memory(*mem_desc, engine, output.mutable_data<scalar_t>());
  auto scale = dnnl::memory(*mem_desc, engine, bn_scale.mutable_data<scalar_t>());
  auto bias = dnnl::memory(*mem_desc, engine, bn_bias.mutable_data<scalar_t>());
  auto out_mean =
          dnnl::memory(*mem_desc, engine, saved_mean.mutable_data<scalar_t>());
  auto out_var =
          dnnl::memory(*mem_desc, engine, saved_inv_var.mutable_data<scalar_t>());

    dnnl::batch_normalization_forward::desc* batch_norm_desc;
  if (train) {
    batch_norm_desc = new dnnl::batch_normalization_forward::desc(
            dnnl::prop_kind::forward_training, *mem_desc, epsilon,
            dnnl::normalization_flags::use_scale_shift);
  } else {
    batch_norm_desc = new dnnl::batch_normalization_forward::desc(
            dnnl::prop_kind::forward_inference, *mem_desc, epsilon,
            dnnl::normalization_flags::use_global_stats);
  }
  auto batch_norm_prim_desc =
          dnnl::batch_normalization_forward::primitive_desc(*batch_norm_desc,
                                                          engine);

  auto batch_norm = dnnl::batch_normalization_forward(batch_norm_prim_desc);
  batch_norm.execute(stream, {{DNNL_ARG_SRC, src},
                              {DNNL_ARG_DST, dst},
                              {DNNL_ARG_MEAN, out_mean},
                              {DNNL_ARG_VARIANCE, out_var},
                              {DNNL_ARG_WEIGHTS, scale},
                              {DNNL_ARG_BIAS, bias}});

  delete mem_desc;
  delete batch_norm_desc;
  /* Tensor & running_mean, Tensor & running_var
  to dealing with moving mean and moving var
  and inv_var
  */
}

void batch_norm_bwd_impl(Tensor& input, Tensor& output_grad, Tensor& bn_scale,
                         Tensor& bn_bias, Tensor& saved_mean,
                         Tensor& saved_inv_var, int mode, double epsilon,
                         Tensor& bn_scale_grad, Tensor& bn_bias_grad,
                         Tensor& input_grad) {
  // sanity check
  auto input_shape = input.dims();

  /* to be finished*/
  // setup
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  using scalar_t = float;

  dnnl::memory::desc* mem_desc;
  if (input_shape.size() == 4) {
    if (mode == HICE_BATCHNORM_PER_ACTIVATION) {
      //          dims =
      //          mkldnn::memory::dims{input_shape[0],input_shape[1]*input_shape[2]*input_shape[3]};
      mem_desc = new dnnl::memory::desc(
          {input_shape[0], input_shape[1] * input_shape[2] * input_shape[3]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 2));
    } else if (mode == HICE_BATCHNORM_SPATIAL) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0], input_shape[1], input_shape[2], input_shape[3]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 4));
    } else {  // mode error
    }
  } else if (input_shape.size() == 5) {
    if (mode == HICE_BATCHNORM_PER_ACTIVATION) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0],
           input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 2));
    } else if (mode == HICE_BATCHNORM_SPATIAL) {
      mem_desc = new dnnl::memory::desc(
          {input_shape[0],
           input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]},
          MKLDNNType<scalar_t>::type,
          hice::get_mkldnn_format_order(LayoutOrder::NCHW, 5));
    } else {  // mode error
    }
  } else {  // dim error
  }

  auto src_input = dnnl::memory(*mem_desc, engine, input.mutable_data<scalar_t>());
  auto src_output =
          dnnl::memory(*mem_desc, engine, output_grad.mutable_data<scalar_t>());
  auto scale = dnnl::memory(*mem_desc, engine, bn_scale.mutable_data<scalar_t>());
  auto bias = dnnl::memory(*mem_desc, engine, bn_bias.mutable_data<scalar_t>());
  auto mean = dnnl::memory(*mem_desc, engine, saved_mean.mutable_data<scalar_t>());
  auto var = dnnl::memory(*mem_desc, engine, saved_inv_var.mutable_data<scalar_t>());
  auto d_scale =
          dnnl::memory(*mem_desc, engine, bn_scale_grad.mutable_data<scalar_t>());
  auto d_bias =
          dnnl::memory(*mem_desc, engine, bn_bias_grad.mutable_data<scalar_t>());
  auto d_input = dnnl::memory(*mem_desc, engine, input_grad.mutable_data<scalar_t>());

    dnnl::batch_normalization_backward::desc* batch_norm_bkw_desc;
  batch_norm_bkw_desc = new dnnl::batch_normalization_backward::desc(
          dnnl::prop_kind::backward, *mem_desc, *mem_desc, epsilon,
      dnnl::normalization_flags::use_scale_shift |
              dnnl::normalization_flags::use_global_stats);

  auto batch_norm_desc = dnnl::batch_normalization_forward::desc(
          dnnl::prop_kind::forward_training, *mem_desc, epsilon,
          dnnl::normalization_flags::use_scale_shift);

  auto batch_norm_prim_desc =
          dnnl::batch_normalization_forward::primitive_desc(batch_norm_desc,
                                                          engine);

  auto batch_norm_bkw_prim_desc =
          dnnl::batch_normalization_backward::primitive_desc(
          *batch_norm_bkw_desc, engine, batch_norm_prim_desc);

  auto batch_norm =
          dnnl::batch_normalization_backward(batch_norm_bkw_prim_desc);
  batch_norm.execute(stream, {{DNNL_ARG_DIFF_DST, src_output},
                              {DNNL_ARG_SRC_ITER, var},
                              {DNNL_ARG_DIFF_SRC, d_input},
                              {DNNL_ARG_DIFF_WEIGHTS, d_scale},
                              {DNNL_ARG_DIFF_BIAS, d_bias},
                              {DNNL_ARG_MEAN, mean},
                              {DNNL_ARG_VARIANCE, var},
                              {DNNL_ARG_WEIGHTS, scale},
                              {DNNL_ARG_BIAS, bias}});

  delete mem_desc;
  delete batch_norm_bkw_desc;
  /* Tensor & running_mean, Tensor & running_var
  to dealing with moving mean and moving var
  and inv_var
  */
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(batch_norm_fwd_dispatcher, &batch_norm_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // output
                     {kCPU, kDense},  // bn_scale
                     {kCPU, kDense},  // bn_bias
                     {kCPU, kDense},  // running_mean
                     {kCPU, kDense},  // running_var
                     {kCPU, kDense},  // saved_mean
                     {kCPU, kDense}   // saved_inv_var
);
HICE_REGISTER_KERNEL(batch_norm_bwd_dispatcher, &batch_norm_bwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // output_grad
                     {kCPU, kDense},  // bn_scale
                     {kCPU, kDense},  // bn_bias
                     {kCPU, kDense},  // saved_mean
                     {kCPU, kDense},  // saved_inv_var
                     {kCPU, kDense},  // bn_scale_grad
                     {kCPU, kDense},  // bn_bias_grad
                     {kCPU, kDense}   // input_grad
);
}  // namespace hice

#endif
