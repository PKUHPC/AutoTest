#ifdef HICE_USE_MKLDNN

#include "hice/device/cpu/context_cpu.h"
#include "hice/device/cpu/allocator_cpu.h"
#include "hice/core/expression_util.h"
#include "hice/basic/reshape.h"
#include "hice/nn/pooling.h"

namespace hice {

namespace {

// infer the expected param from user's input.
inline static std::vector<int64_t> infer_params(
  ConstIntArrayRef origin_param,
  int64_t ndim_pooling, 
  int64_t default_value) {
  auto length = origin_param.size();
  if (length == ndim_pooling) {
    return std::vector<int64_t>(origin_param.begin(), origin_param.end());
  } else if (length == 0) {
    return std::vector<int64_t>(ndim_pooling, default_value);
  } else {
    HICE_CHECK(length == 1);
    return std::vector<int64_t>(ndim_pooling, origin_param[0]);
  }
}

// AVG Forward
template<typename scalar_t>
void pooling_avg_fwd_kernel(const Tensor& input, ConstIntArrayRef kernel,
                            ConstIntArrayRef stride, ConstIntArrayRef padding, 
                            Tensor& output) {
  // pre-definition
  const int N_DIM = input.ndim();
  LayoutOrder LAYOUT_ORDER = LayoutOrder::NCHW;
  // data prepare for mkl-dnn
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  auto dims_input = input.dims();
  auto dims_output = output.dims();
  // memory for input
  auto pool_src_tz = dnnl::memory::dims(dims_input.begin(), dims_input.end());
  auto mem_src_desc = dnnl::memory::desc(pool_src_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_src = dnnl::memory(mem_src_desc, engine, const_cast<Tensor &>(input).mutable_data<scalar_t>());
  // memory for output
  auto pool_dst_tz = dnnl::memory::dims(dims_output.begin(), dims_output.end());
  auto mem_dst_desc = dnnl::memory::desc(pool_dst_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_dst = dnnl::memory(mem_dst_desc, engine, output.mutable_data<scalar_t>());
  // desc for pooling
  auto pool_kernel = dnnl::memory::dims(kernel.begin(), kernel.end());
  auto pool_strides = dnnl::memory::dims(stride.begin(), stride.end());
  auto pool_padding = dnnl::memory::dims(padding.begin(), padding.end());
  auto pool_desc =
          dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,
                                      dnnl::algorithm::pooling_avg_include_padding, // pooling_avg_exclude_padding
                                    mem_src_desc, mem_dst_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
#if 0
  auto print_v = [](ConstIntArrayRef arr){
    for (int i=0;i<arr.size(); ++i){
      std::cout<<arr[i]<<",";
    }
      std::cout<<std::endl;
  };
  std::cout << "dims_input=";
  print_v(dims_input);
  std::cout << "dims_output=";
  print_v(dims_output);
  std::cout << "kernel=";
  print_v(kernel);
  std::cout << "stride_valid=";
  print_v(stride_valid);
  std::cout << "padding_valid=";
  print_v(padding_valid);
#endif
  // desc for pooling primitive
  auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine);
  auto pooling = dnnl::pooling_forward(pool_prim_desc);
  // call mkl-dnn
  pooling.execute(stream, {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_DST, mem_dst}});
}

// AVG Backward
template<typename scalar_t>
void pooling_avg_bwd_kernel(const Tensor& input, const Tensor& grad_output,
                            ConstIntArrayRef kernel, ConstIntArrayRef stride, 
                            ConstIntArrayRef padding, Tensor& grad_input) {
  // pre-definition
  const int N_DIM = input.ndim();
  LayoutOrder LAYOUT_ORDER = LayoutOrder::NCHW;
  // data prepare for mkl-dnn
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  ConstIntArrayRef dims_grad_output = grad_output.dims();
  ConstIntArrayRef dims_grad_input = grad_input.dims();
  // memory for grad_output
  auto pool_grad_output_tz = dnnl::memory::dims(dims_grad_output.begin(), dims_grad_output.end());
  auto mem_grad_output_desc = dnnl::memory::desc(pool_grad_output_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_grad_output = dnnl::memory(mem_grad_output_desc, engine, const_cast<Tensor &>(grad_output).mutable_data<scalar_t>());
  // memory for grad_input
  auto pool_grad_input_tz = dnnl::memory::dims(dims_grad_input.begin(), dims_grad_input.end());
  auto mem_grad_input_desc = dnnl::memory::desc(pool_grad_input_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_grad_input = dnnl::memory(mem_grad_input_desc, engine, grad_input.mutable_data<scalar_t>());
  // desc for pooling
  auto pool_kernel = dnnl::memory::dims(kernel.begin(), kernel.end());
  auto pool_strides = dnnl::memory::dims(stride.begin(), stride.end());
  auto pool_padding = dnnl::memory::dims(padding.begin(), padding.end());
  // auto print_v = [](ConstIntArrayRef arr){
  //   for (int i=0;i<arr.size(); ++i){
  //     std::cout<<arr[i]<<",";
  //   }
  //     std::cout<<std::endl;
  // };
  // std::cout << "dims_grad_output=";
  // print_v(dims_grad_output);
  // std::cout << "dims_grad_input=";
  // print_v(dims_grad_input);
  // std::cout << "kernel=";
  // print_v(kernel);
  // std::cout << "stride_valid=";
  // print_v(stride_valid);
  // std::cout << "padding_valid=";
  // print_v(padding_valid);
  auto pool_fwd_desc =
          dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,
                                      dnnl::algorithm::pooling_avg_include_padding,
                                    mem_grad_input_desc, mem_grad_output_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
  auto pool_bwd_desc =
          dnnl::pooling_backward::desc(dnnl::algorithm::pooling_avg_include_padding,
                                    mem_grad_input_desc, mem_grad_output_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
  // desc for pooling primitive
  auto pool_fwd_prim_desc = dnnl::pooling_forward::primitive_desc(pool_fwd_desc, engine);
  auto pool_bwd_prim_desc = dnnl::pooling_backward::primitive_desc(pool_bwd_desc, engine, pool_fwd_prim_desc);
  auto pooling_bwd = dnnl::pooling_backward(pool_bwd_prim_desc);
  // call mkl-dnn
  pooling_bwd.execute(stream, {{DNNL_ARG_DIFF_DST, mem_grad_output},
                               {DNNL_ARG_DIFF_SRC, mem_grad_input}});
}

// MAX Forward
template<typename scalar_t>
void pooling_max_fwd_kernel(const Tensor& input, ConstIntArrayRef kernel,
                            ConstIntArrayRef stride, ConstIntArrayRef padding, 
                            Tensor& indices, Tensor& output) {
  // pre-definition
  const int N_DIM = input.ndim();
  LayoutOrder LAYOUT_ORDER = LayoutOrder::NCHW;
  // data prepare for mkl-dnn
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  auto dims_input = input.dims();
  auto dims_output = output.dims();
  // memory for input
  auto pool_src_tz = dnnl::memory::dims(dims_input.begin(), dims_input.end());
  auto mem_src_desc = dnnl::memory::desc(pool_src_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_src = dnnl::memory(mem_src_desc, engine, const_cast<Tensor &>(input).mutable_data<scalar_t>());
  // memory for output
  auto pool_dst_tz = dnnl::memory::dims(dims_output.begin(), dims_output.end());
  auto mem_dst_desc = dnnl::memory::desc(pool_dst_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_dst = dnnl::memory(mem_dst_desc, engine, output.mutable_data<scalar_t>());
  // desc for pooling
  auto pool_kernel = dnnl::memory::dims(kernel.begin(), kernel.end());
  auto pool_strides = dnnl::memory::dims(stride.begin(), stride.end());
  auto pool_padding = dnnl::memory::dims(padding.begin(), padding.end());
  auto pool_desc =
          dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,
                                      dnnl::algorithm::pooling_max, //
                                    mem_src_desc, mem_dst_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
  // desc for pooling primitive
  auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine);
  auto pooling = dnnl::pooling_forward(pool_prim_desc);
  // memory for workspace (for saving indices)
  auto mem_ws_desc = pool_prim_desc.workspace_desc();
  auto mem_ws = dnnl::memory(mem_ws_desc, engine);
  // call mkl-dnn
  pooling.execute(stream, {{DNNL_ARG_SRC, mem_src},
                           {DNNL_ARG_DST, mem_dst},
                           {DNNL_ARG_WORKSPACE, mem_ws}});
  // The following block is to read indices from mem_ws.
  {
    // NOTE:
    // Mkl-dnn does not give API to get datatype of mem_desc,
    // so here we do a deduction. Then read it into our Tensor.
    auto size_indices = indices.size();
    auto size_in_byte_ws = mem_ws_desc.get_size();
    auto sizeof_type_ws = size_indices / size_in_byte_ws * 8;
    HICE_CHECK(size_indices % size_in_byte_ws == 0);
    HICE_CHECK(indices.scalar_type() == kInt32);
    auto ptr_mem_ws = mem_ws.get_data_handle();
    auto ptr_indices = indices.mutable_data<int32_t>();
    switch(sizeof_type_ws) {
      case 8: {
        using sc_type_ws = unsigned char;/* uint8 */
        for (int i = 0; i < size_indices; ++i) {
          ptr_indices[i] = ((sc_type_ws *)ptr_mem_ws)[i];
        }
        break;
      }
      case 32: {
        using sc_type_ws = int;
        for (int i = 0; i < size_indices; ++i) {
          ptr_indices[i] = ((sc_type_ws *)ptr_mem_ws)[i];
        }
        break;
      }
      default:
        HICE_LOG(ERROR) << "Type infer failed when save indices from mkl-dnn memory. "
          << "size_of_type for mkl-dnn memory equals to " << sizeof_type_ws
          << ". This bug occurs because the mkl-dnn being used is not the"
          << "expected version for pooling-max. It would work well with"
          << "mkl-dnn-v1.0-pc2";
    } // switch end
  } // block end
} // function end


// MAX Backward
template<typename scalar_t>
void pooling_max_bwd_kernel(const Tensor& input, const Tensor& grad_output,
                            const Tensor& indices, 
                            ConstIntArrayRef kernel, ConstIntArrayRef stride, 
                            ConstIntArrayRef padding, Tensor& grad_input) {
  // pre-definition
  const int N_DIM = input.ndim();
  LayoutOrder LAYOUT_ORDER = LayoutOrder::NCHW;
  // data prepare for mkl-dnn
  CPUContext cpu_ctx;
  auto engine = cpu_ctx.mkldnn_engine();
  auto stream = cpu_ctx.mkldnn_stream();
  ConstIntArrayRef dims_grad_output = grad_output.dims();
  ConstIntArrayRef dims_grad_input = grad_input.dims();
  // memory for grad_output
  auto pool_grad_output_tz = dnnl::memory::dims(dims_grad_output.begin(), dims_grad_output.end());
  auto mem_grad_output_desc = dnnl::memory::desc(pool_grad_output_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_grad_output = dnnl::memory(mem_grad_output_desc, engine, const_cast<Tensor &>(grad_output).mutable_data<scalar_t>());
  // memory for indices
  // auto pool_grad_output_tz = mkldnn::memory::dims(dims_grad_output.begin(), dims_grad_output.end());
  // auto mem_grad_output_desc = mkldnn::memory::desc(pool_grad_output_tz, MKLDNNType<scalar_t>::type,
  //                         hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  // auto mem_grad_output = mkldnn::memory(mem_grad_output_desc, engine, grad_output.data<scalar_t>());
  // memory for grad_input
  auto pool_grad_input_tz = dnnl::memory::dims(dims_grad_input.begin(), dims_grad_input.end());
  auto mem_grad_input_desc = dnnl::memory::desc(pool_grad_input_tz, MKLDNNType<scalar_t>::type,
                          hice::get_mkldnn_format_order(LAYOUT_ORDER, N_DIM));
  auto mem_grad_input = dnnl::memory(mem_grad_input_desc, engine, grad_input.mutable_data<scalar_t>());
  // desc for pooling
  auto pool_kernel = dnnl::memory::dims(kernel.begin(), kernel.end());
  auto pool_strides = dnnl::memory::dims(stride.begin(), stride.end());
  auto pool_padding = dnnl::memory::dims(padding.begin(), padding.end());
  auto pool_fwd_desc =
          dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,
                                      dnnl::algorithm::pooling_max,
                                    mem_grad_input_desc, mem_grad_output_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
  auto pool_bwd_desc =
          dnnl::pooling_backward::desc(dnnl::algorithm::pooling_max,
                                    mem_grad_input_desc, mem_grad_output_desc, pool_strides,
                                    pool_kernel, pool_padding, pool_padding);
  // desc for pooling primitive
  auto pool_fwd_prim_desc = dnnl::pooling_forward::primitive_desc(pool_fwd_desc, engine);
  auto pool_bwd_prim_desc = dnnl::pooling_backward::primitive_desc(pool_bwd_desc, engine, pool_fwd_prim_desc);
  auto pooling_bwd = dnnl::pooling_backward(pool_bwd_prim_desc);
  // memory for workspace (for saving indices)
  auto mem_ws_desc = pool_fwd_prim_desc.workspace_desc();
  void* ptr_mem_ws;
  // The following block is to load indices to mem_ws.
  {
    // NOTE:
    // Mkl-dnn does not give API to get datatype of mem_desc,
    // so here we do a deduction. Then read it into our Tensor.
    auto size_indices = indices.size();
    auto size_in_byte_ws = mem_ws_desc.get_size();
    auto sizeof_type_ws = size_indices / size_in_byte_ws * 8;
    HICE_CHECK(size_indices % size_in_byte_ws == 0);
    HICE_CHECK(indices.scalar_type() == kInt32);
    auto ptr_indices = indices.data<int32_t>();
    switch(sizeof_type_ws) {
      case 8: {
        using sc_type_ws = unsigned char;/* uint8 */
        ptr_mem_ws = cpu_allocator()->allocate_raw(sizeof(sc_type_ws) * size_indices);
        HICE_CHECK(ptr_mem_ws!=NULL);
        for (int i = 0; i < size_indices; ++i) {
          ((sc_type_ws *)ptr_mem_ws)[i] = ptr_indices[i];
        }
        break;
      }
      case 32: {
        using sc_type_ws = int;
        ptr_mem_ws = cpu_allocator()->allocate_raw(sizeof(sc_type_ws) * size_indices);
        HICE_CHECK(ptr_mem_ws!=NULL);
        for (int i = 0; i < size_indices; ++i) {
          ((sc_type_ws *)ptr_mem_ws)[i] = ptr_indices[i];
        }
        break;
      }
      default:
        HICE_LOG(ERROR) << "Type infer failed when load mkl-dnn memory from indices, "
          << "size_of_type for mkl-dnn memory equals to " << sizeof_type_ws
          << ". This bug occurs because the mkl-dnn being used is not the"
          << "expected version for pooling-max. It would work well with"
          << "mkl-dnn-v1.0-pc2";
    } // switch end
  } // block end
  auto mem_ws = dnnl::memory(mem_ws_desc, engine, ptr_mem_ws);
  // call mkl-dnn
  pooling_bwd.execute(stream, {{DNNL_ARG_DIFF_DST, mem_grad_output},
                               {DNNL_ARG_DIFF_SRC, mem_grad_input},
                               {DNNL_ARG_WORKSPACE, mem_ws}});
  cpu_allocator()->deallocate_raw(ptr_mem_ws);
}



void pooling_avg_fwd_impl(const Tensor& input, ConstIntArrayRef kernel,
                            ConstIntArrayRef stride, ConstIntArrayRef padding, 
                            Tensor& output) {
  // std::cout << "In mkldnn pooling_avg_fwd_impl"<< std::endl;
  HICE_CHECK(input.is_default_layout());
  HICE_CHECK(input.scalar_type() == kFloat);
  int64_t ndim = input.ndim();
  pooling_avg_fwd_kernel<float>(input, kernel, stride, 
                                padding, output);
}

void pooling_avg_bwd_impl(const Tensor& input, const Tensor& output,
                          const Tensor& grad_output,
                          ConstIntArrayRef kernel, ConstIntArrayRef stride, 
                          ConstIntArrayRef padding, Tensor& grad_input) {
  // std::cout << "In mkldnn pooling_avg_bwd_impl"<< std::endl;
  HICE_CHECK(input.is_default_layout());
  HICE_CHECK(output.is_default_layout());
  HICE_CHECK(grad_output.is_default_layout());
  HICE_CHECK(input.scalar_type() == kFloat);
  HICE_CHECK(grad_output.scalar_type() == kFloat);
  pooling_avg_bwd_kernel<float>(input, grad_output, kernel, 
                                stride, padding, grad_input);
}

void pooling_max_fwd_impl(const Tensor& input, ConstIntArrayRef kernel,
                          ConstIntArrayRef stride, ConstIntArrayRef padding, 
                          Tensor& indices, Tensor& output, bool resizable) {
  // std::cout << "In mkldnn pooling_max_fwd_impl"<< std::endl;
  HICE_CHECK(input.is_default_layout());
  HICE_CHECK(input.scalar_type() == kFloat);
  int64_t ndim = input.ndim();
  // check and infer params
  const int DIM_POOLING = ndim - 2;
  const int DEFAULT_STRIDE = 1;
  const int DEFAULT_PADDING = 0;
  // kernel != null
  HICE_CHECK(kernel.size() == DIM_POOLING || kernel.size() == 1);
  std::vector<int64_t> kernel_new
          = infer_params(kernel, DIM_POOLING, -1);
  std::vector<int64_t> stride_new
          = infer_params(stride, DIM_POOLING, DEFAULT_STRIDE);
  std::vector<int64_t> padding_new
          = infer_params(padding, DIM_POOLING, DEFAULT_PADDING);
  // resize output and indices
  ConstIntArrayRef dims_input = input.dims();
  std::vector<int64_t> dims_output({/* batch= */dims_input[0], 
                                    /* channel= */dims_input[1]});
  for (int i = 2; i < ndim; ++i) {
    auto sz_ipt = dims_input[i];
    auto sz_krnl = kernel_new[i - 2];
    auto sz_strd = stride_new[i - 2];
    auto sz_pad = padding_new[i - 2];
    auto sz_otpt = (sz_ipt + 2 * sz_pad - sz_krnl) / sz_strd + 1;
    dims_output.push_back(sz_otpt);
  }
  ExpressionUtil::may_resize_result(output, dims_output, resizable);
  ExpressionUtil::may_resize_result(indices, dims_output, resizable);
  // convert 1d pooling into 2d pooling
  if (ndim == 3) {
    kernel_new.push_back(1);
    stride_new.push_back(1);
    padding_new.push_back(0);
    Tensor input_new = expand_dims(input, -1);
    Tensor output_new = expand_dims(output, -1);
    Tensor indices_new = expand_dims(indices, -1);
    pooling_max_fwd_kernel<float>(input_new, kernel_new, stride_new, 
                                  padding_new, indices_new, output_new);
  } else if (ndim == 4 || ndim == 5) {
    pooling_max_fwd_kernel<float>(input, kernel_new, stride_new, padding_new,
                                  indices, output);
  } else {
    HICE_LOG(ERROR) << "Unsupported dimension in pooling.";
  }
}

void pooling_max_bwd_impl(const Tensor& input, const Tensor& output,
                          const Tensor& grad_output, const Tensor& indices,
                          ConstIntArrayRef kernel, ConstIntArrayRef stride, 
                          ConstIntArrayRef padding, Tensor& grad_input, 
                          bool resizable) {
  // std::cout << "In mkldnn pooling_max_bwd_impl"<< std::endl;
  HICE_CHECK(input.is_default_layout());
  HICE_CHECK(output.is_default_layout());
  HICE_CHECK(grad_output.is_default_layout());
  HICE_CHECK(output.dims() == grad_output.dims());
  int64_t ndim = input.ndim();
  // check params
  const int DIM_POOLING = ndim - 2;
  const int DEFAULT_STRIDE = 1;
  const int DEFAULT_PADDING = 0;
  // kernel != null
  HICE_CHECK(kernel.size() == DIM_POOLING || kernel.size() == 1);
  std::vector<int64_t> kernel_new
          = infer_params(kernel, DIM_POOLING, -1);
  std::vector<int64_t> stride_new
          = infer_params(stride, DIM_POOLING, DEFAULT_STRIDE);
  std::vector<int64_t> padding_new
          = infer_params(padding, DIM_POOLING, DEFAULT_PADDING);
  // resize grad_input
  ExpressionUtil::may_resize_result(grad_input, input.dims(), resizable);
  // convert 1d pooling into 2d pooling
  if (ndim == 3) {
    kernel_new.push_back(1);
    stride_new.push_back(1);
    padding_new.push_back(0);
    Tensor input_new = expand_dims(input, -1);
    Tensor indices_new = expand_dims(indices, -1);
    Tensor grad_output_new = expand_dims(grad_output, -1);
    Tensor grad_input_new = expand_dims(grad_input, -1);
    pooling_max_bwd_kernel<float>(input_new, grad_output_new, indices_new,
                                  kernel_new, stride_new, padding_new, 
                                  grad_input_new);
  } else if (ndim == 4 || ndim == 5) {
    // output is not used in mkl-dnn
    pooling_max_bwd_kernel<float>(input, grad_output, indices, 
                                  kernel_new, stride_new, padding_new, 
                                  grad_input);
  } else {
    HICE_LOG(ERROR) << "Unsupported dimension in pooling.";
  }
}

} // namespace anonymous

HICE_REGISTER_KERNEL(
  pooling_avg_fwd_dispatcher, 
  &pooling_avg_fwd_impl, 
  {kCPU, kDense}, // first operand
  {kCPU, kDense} // result 
);

HICE_REGISTER_KERNEL(
  pooling_avg_bwd_dispatcher, 
  &pooling_avg_bwd_impl, 
  {kCPU, kDense}, // input
  {kCPU, kDense}, // output
  {kCPU, kDense}, // grad_output
  {kCPU, kDense} // grad_input
);

HICE_REGISTER_KERNEL(
  pooling_max_fwd_dispatcher, 
  &pooling_max_fwd_impl, 
  {kCPU, kDense}, // input
  {kCPU, kDense}, // indices
  {kCPU, kDense} // output 
);

HICE_REGISTER_KERNEL(
  pooling_max_bwd_dispatcher, 
  &pooling_max_bwd_impl, 
  {kCPU, kDense}, // input
  {kCPU, kDense}, // output
  {kCPU, kDense}, // grad_output
  {kCPU, kDense}, // indices
  {kCPU, kDense} // grad_input 
);

} // namespace hice

#endif
