#pragma once

#ifndef HICE_USE_MKLDNN
#error("Since HICE is not built with mkldnn, so you should not include this file.");
#endif

#include <dnnl.hpp>

#include "hice/core/layout.h"
#include "hice/core/scalar_type.h"
#include "hice/util/loguru.h"

namespace hice {
  
template <typename T>
struct MKLDNNType;

template <>
struct MKLDNNType<float> {
  static const dnnl::memory::data_type type = dnnl::memory::data_type::f32;
};

template <>
struct MKLDNNType<int32_t> {
  static const dnnl::memory::data_type type = dnnl::memory::data_type::s32;
};

template <>
struct MKLDNNType<int8_t> {
  static const dnnl::memory::data_type type = dnnl::memory::data_type::s8;
};

template <>
struct MKLDNNType<uint8_t> {
  static const dnnl::memory::data_type type = dnnl::memory::data_type::u8;
};

inline dnnl::memory::data_type get_mkldnn_data_type(ScalarType scalar_type) {
  if (scalar_type == kFloat) {
    return dnnl::memory::data_type::f32;
  } else if (scalar_type == kInt32) {
    return dnnl::memory::data_type::s32;
  }
  std::string msg("get_mkldnn_data_type() not supported for ");
  msg += to_string(scalar_type);
  throw std::runtime_error(msg);
}

inline dnnl::memory::format_tag
get_mkldnn_format_order(const LayoutOrder& order, int ndim) {
    dnnl::memory::format_tag mkldnn_order;
  switch (ndim) {
    case 1:
      mkldnn_order = dnnl::memory::format_tag::x;
      break;
    case 2:
      mkldnn_order = dnnl::memory::format_tag::nc;
      break;
    case 3:
      if (order == LayoutOrder::NCHW) {
        mkldnn_order = dnnl::memory::format_tag::ncw;
      } else {
        mkldnn_order = dnnl::memory::format_tag::nwc;
      }
      break;
    case 4:
      if (order == LayoutOrder::NCHW) {
        mkldnn_order = dnnl::memory::format_tag::nchw;
      } else {
        mkldnn_order = dnnl::memory::format_tag::nhwc;
      }
      break;
    case 5:
      if (order == LayoutOrder::NCHW) {
        mkldnn_order = dnnl::memory::format_tag::ncdhw;
      } else {
        mkldnn_order = dnnl::memory::format_tag::ndhwc;
      }
      break;
    default:
      HICE_LOG(FATAL) << "Input ndim must be <= 5 and >=1"
                      << ndim;
  }
  return mkldnn_order;
}

inline dnnl::memory::format_tag get_mkldnn_format_tag(int64_t ndim,
                                                        int64_t g = 1) {
    dnnl::memory::format_tag tag;
  switch (ndim) {
    case 1:
      tag = dnnl::memory::format_tag::x;
      break;
    case 2:
      tag = dnnl::memory::format_tag::nc;
      break;
    case 3:
      tag = g == 1 ? dnnl::memory::format_tag::oiw
                   : dnnl::memory::format_tag::goiw;
      break;
    case 4:
      tag = g == 1 ? dnnl::memory::format_tag::oihw
                   : dnnl::memory::format_tag::goihw;
      break;
    case 5:
      tag = g == 1 ? dnnl::memory::format_tag::oidhw
                   : dnnl::memory::format_tag::goidhw;
      break;
    default:
      HICE_LOG(FATAL) << "Input ndim must be <= 5 and >=1"
                      << ndim;
  }
  return tag;
}

} // namespace hice