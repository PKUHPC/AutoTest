#pragma once

#include <numeric>
#include <vector>
#include "hice/core/dimension.h"
#include "hice/core/tensor.h"

namespace hice {

enum class ExprType : int16_t { 
  kUnary = 10, 
  kBinary = 11, 
  kReduction = 12 
};

class Expression {
 private:
  using TensorArray = std::vector<Tensor>;
  using StridesArray = std::vector<std::vector<int64_t>>;

 public:
  TensorArray& outputs() { return outputs_; }

  Tensor& output(int i) { return outputs_[i]; }

  Tensor& reviewed_output(int i) { return reviewed_outputs_[i]; }

  const Tensor& input(int i) { return inputs_[i]; }

  const TensorArray& inputs() { return inputs_; }

  int num_inputs() { return inputs_.size(); }

  int num_outputs() { return outputs_.size(); }

  std::vector<int64_t>& strides_input(int i) { return strides_inputs_[i]; }

  std::vector<int64_t>& strides_output(int i) { return strides_outputs_[i]; }

  void set_type(ExprType type) { type_ = type; }

  void set_resizable(bool resizable) { output_resizable_ = resizable; }

  void add_input(const Tensor& in) {
    this->inputs_.emplace_back(in);
  }
  void add_output(Tensor& out) {
    this->outputs_.emplace_back(out);
  }

  // resize output for unary and binary
  void resize_output();

  // resize output for reduction
  void resize_output(std::vector<bool> mask, bool keep_dim);

  // Only used in reduction for special case 'keepdim = false'
  void review_reduction_result(std::vector<bool> mask, bool keep_dim);

  // Only used in binary and reduction expression.
  // The length of strides for inputs and output equals to max(dim_in, dim_out).
  void prepare_strides();

  // Create a mask bool value array (ndim) which represents whether this
  // dimension will be reduced or not.
  std::vector<bool> make_reduction_mask(ConstIntArrayRef reduced_dims);

  // Move reduction dims to the front, sort the other dims
  // in ascending order, doesn't change the remain underlayer storage
  std::vector<int64_t> reorder_dims(std::vector<int64_t> strides);

  std::vector<int64_t> permute_dims(const std::vector<int64_t> in,
                                    std::vector<int64_t> perm);

 private:
  ExprType type_;
  TensorArray inputs_;
  TensorArray outputs_;
  TensorArray reviewed_outputs_;
  StridesArray strides_inputs_;
  StridesArray strides_outputs_;
  bool output_resizable_ = false;
};
}  // namespace hice
