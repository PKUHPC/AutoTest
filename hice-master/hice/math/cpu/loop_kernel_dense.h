#pragma once
#include "hice/basic/cpu/index_helper.h"
#include "hice/math/cpu/vectorization/vec256.h"

namespace hice {

namespace {
template <typename TScalarType, typename TArrayType>
TScalarType* compute_offset(TScalarType* base_ptr, TArrayType& location,
                            TArrayType& strides) {
  TScalarType* ptr = base_ptr;
  for (int i = 0; i < location.size(); i++) {
    ptr += location[i] * strides[i];
  }
  return ptr;
}
}

template <typename TScalarType1, typename TScalarType2, typename TOp>
void serial_unary_loop_kernel(const TScalarType1* in, 
                              TScalarType2* out, 
                              IndexHelper idx_hlpr_tensor,
                              IndexHelper idx_hlpr_result,
                              int64_t start, int64_t end, TOp op) {
  int64_t index = start;
  while (index < end) {
    auto offset_in = idx_hlpr_tensor.linear_index_to_offset(index);
    auto offset_out = idx_hlpr_result.linear_index_to_offset(index);
    out[offset_out] = op(in[offset_in]);
    // std::cout<<"index="<<index;
    // std::cout<<", offset_in="<<offset_in;
    // std::cout<<", offset_out="<<offset_out<<std::endl;
    index++;
  }
}

template <typename TScalarType, typename TOp, typename TVOp>
void serial_unary_loop_kernel_vec(const TScalarType* in, 
                                  TScalarType* out, 
                                  int64_t start, int64_t end, 
                                  TOp op, TVOp vec_op) {
  using Vec = Vec256<TScalarType>;
  
  int64_t index = start;
  while (index < end - Vec::size()) {
    auto vec_out = vec_op(Vec::loadu(in + index));
    vec_out.store(out + index);
    index += Vec::size();
  }
  while (index < end) {
    out[index] = op(in[index]);
    index++;
  }
}

template <typename TScalarType, typename TOp>
void serial_unary_loop_kernel_basic(const TScalarType* in, 
                                    TScalarType* out, 
                                    int64_t start, int64_t end, 
                                    TOp op) {
  int64_t index = start;
  while (index < end) {
    out[index] = op(in[index]);
    index++;
  }
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, typename TOp>
void serial_binary_loop_kernel(const TScalarType1* in1, 
                               const TScalarType2* in2,
                               TScalarType3* out, 
                               IndexHelper idx_hlpr_in1,
                               IndexHelper idx_hlpr_in2,
                               IndexHelper idx_hlpr_out,
                               int64_t start, int64_t end, TOp op) {
  int64_t index = start;
  while (index < end) {
    auto offset_in1 = idx_hlpr_in1.linear_index_to_offset(index);
    auto offset_in2 = idx_hlpr_in2.linear_index_to_offset(index);
    auto offset_out = idx_hlpr_out.linear_index_to_offset(index);
    out[offset_out] = op(in1[offset_in1], in2[offset_in2]);
    // std::cout<<"index="<<index;
    // std::cout<<",offset_in1="<<offset_in1;
    // std::cout<<",offset_in2="<<offset_in2;
    // std::cout<<",offset_out="<<offset_out<<std::endl;
    index++;
  }
}

template <typename TScalarType, typename TOp>
void serial_binary_loop_kernel_basic(const TScalarType* in1, 
                                  const TScalarType* in2,
                                  TScalarType* out, 
                                  int64_t start, int64_t end, 
                                  TOp op) {
  int64_t index = start;
  while (index < end) {
    out[index] = op(in1[index], in2[index]);
    index++;
  }
}

template <typename TScalarType, typename TOp, typename TVOp>
void serial_binary_loop_kernel_vec(const TScalarType* in1, 
                                  const TScalarType* in2,
                                  TScalarType* out, 
                                  int64_t start, int64_t end, 
                                  TOp op, TVOp vec_op) {
  using Vec = Vec256<TScalarType>;
 
  int64_t index = start;
  int64_t num_serial = (end - start) % Vec::size();
  while (index < end - num_serial) {
    auto vec_out = vec_op(Vec::loadu(in1 + index), Vec::loadu(in2 + index));
    vec_out.store(out + index);
    index += Vec::size();
  }
  while (index < end) {
    out[index] = op(in1[index], in2[index]);
    index++;
  }
}

template <typename TScalarType, typename TArrayType, typename TOp>
void reduce_kernel(const TScalarType* base_ptr_in, TScalarType* base_ptr_out,
                   TArrayType& strides_in, TArrayType& strides_out,
                   TArrayType& dims, int num_items, TOp op) {
  if (dims.size() <= 1) {
    for (int i = 0; i < dims[0]; i++) {
      auto in_ptr = base_ptr_in + i;
      *base_ptr_out = op(*base_ptr_out, *in_ptr);
    }
  } else {
    int count = 0;
    TArrayType step_value(dims.size(), 0);
    while (count < num_items) {
      auto in = compute_offset(base_ptr_in, step_value, strides_in);
      auto out = compute_offset(base_ptr_out, step_value, strides_out);
      for (int i = 0; i < dims[1]; i++) {
        for (int j = 0; j < dims[0]; j++) {
          auto in_ptr = in + j * strides_in[0];
          auto out_ptr = out + j * strides_out[0];
          *out_ptr = op(*out_ptr, *in_ptr);
        }
        in = in + strides_in[1];
        out = out + strides_out[1];
      }
      count += dims[1] * dims[0];
      for (int i = 2; i < dims.size(); i++) {
        if (step_value[i] < (dims[i] - 1)) {
          step_value[i]++;
          break;
        }
        step_value[i] = 0;
      }
    }
  }
}

} // namespace hice
