#pragma once

#include "expression.h"

namespace hice {

class ExpressionUtil {
public:
  // Make unary expression
  // resizable: whether the output tensor could be resized.
  static Expression make_unary_expr(const Tensor& in, Tensor& out,
                                    bool resizable);

  // Make binary expression
  // resizable: whether the output tensor could be resized.
  static Expression make_binary_expr(const Tensor& in1, const Tensor& in2,
                                      Tensor& out, bool resizable);

  // Make reduction expression
  // resizable: whether the output tensor could be resized.
  static Expression make_reduction_expr(const Tensor& in, Tensor& out,
                                        ConstIntArrayRef reduced_dims,
                                        bool keep_dim, bool resizable);

  // Resize tensor with given dims. 
  // if resizable = true, output would be resized with dims_output
  // if resizable = false, the dims of output should be same with given dims.
  static void may_resize_result(Tensor& output, ConstIntArrayRef dims_output,
                                bool resizable);

  // Rules: loop to check if a certain dimension in tensor is equal to 1,
  // change the stride of this dimension to 0
  static std::vector<int64_t> strides_for_computing(
    ConstIntArrayRef strides_old, ConstIntArrayRef dims,
    int64_t length_stride);

  // return a longer array info about minor_to_major
  // eg. minor_to_major_old =       [3, 1, 2, 0], length = 6,
  //     minor_to_major_new = [5, 4, 3, 1, 2, 0]
  static std::vector<int64_t> minor_to_major_for_computing(
    ConstIntArrayRef minor_to_major_old, int64_t length);

 private:
  HICE_DISABLE_COPY_AND_ASSIGN(ExpressionUtil);
};

}