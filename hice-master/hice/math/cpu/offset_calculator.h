#pragma once

#include <vector>

namespace hice {

// NOTE: Different from OffsetCalculator on cuda, it is not for
// reduce expression. It is only used in broadcasting case.
struct OffsetCalculator {
  
  // dims: dims of output(broadcasted from dims of inputs)
  // strides_in: strides_for_computing of input
  OffsetCalculator(ConstIntArrayRef dimensions,
                   ConstIntArrayRef strides, 
                   ConstIntArrayRef minor_to_major) 
                   : dims_(dimensions),
                     strides_(strides.begin(), strides.end()),
                     minor_to_major_(minor_to_major.begin(), minor_to_major.end()){ 
  }

  // param idx: index for output
  // basic idea: 
  // 1. idx -> loc(location under dims)
  // 2. offset = sum(loc * strides)
  int64_t idx2offset(int64_t idx) const {
    int64_t offset = 0;
    int64_t location_i = 0;
#pragma unroll
    for (int64_t i = dims_.size() - 1; i >= 0 ; i--) {
      // int64_t ii = minor_to_major_[i];
      int64_t ii = i;
      location_i = idx % dims_[ii];
      idx /= dims_[ii];
      offset += location_i * strides_[ii];
      // std::cout<<"idx="<<idx;
      // std::cout<<",i="<<i;
      // std::cout<<",ii="<<ii;
      // std::cout<<",location_i="<<location_i;
      // std::cout<<",dims_[ii]="<<dims_[ii];
      // std::cout<<",strides_[ii]="<<strides_[ii];
      // std::cout<<std::endl;
    }
    return offset;
  }

private:
  ConstIntArrayRef dims_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> minor_to_major_;
}; // struct OffsetCalculator


}