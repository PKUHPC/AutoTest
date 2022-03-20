#pragma once

struct OffsetCalculator {
  static constexpr int MAX_DIMS_NUM = 25;
  
  OffsetCalculator(int ndim, const int64_t *dims, const int64_t *strides)
      : ndim(ndim) {
    for (int i = 0; i < MAX_DIMS_NUM; i++) {
      dims_[i] = (i < ndim) ? dims[i] : 1;
      strides_[i] = (i < ndim) ? strides[i] : 0;
    }
  }

  __device__ int idx2offset(int idx) const {
    // mod_value: loc[i]
    // rem: index from loc[0:i] - loc[i]
    // prod: strides[i] of dims_
    // offset: index for input
    // eg. dims_=[3,4,2],strides_=[0,2,1],loc=[2,3,1],i=1
    //     => mod_value = loc[1] 
    //                  = 3
    //     => rem = index(loc[0:1]) - loc[1]
    //            = (2*4+3) - 3
    //     => prod = strides_for_dims(8,2,1)[1]
    //             = 2
    //     => offset = offset + mod_value * strides_[1]
    //               = offset + 3*2
    int offset = 0;
    int rem = idx;
    int prod = 1;
#pragma unroll
    for(int i = 0;i < ndim; i++){
      int mod_value = (rem / prod) % dims_[i];
      rem = idx - mod_value * prod;
      prod = prod * dims_[i];
      offset += mod_value * strides_[i];
    }
    // for (int i = 0; i < MAX_DIMS_NUM; i++) {
    //   if (i == ndim) {
    //     break;
    //   }
    //   int mod_value = (rem / prod) % dims_[i];
    //   rem = idx - mod_value * prod;
    //   prod = prod * dims_[i];
    //   offset += mod_value * strides_[i];
    // }
    return offset;
  }

  int ndim;
  int64_t dims_[MAX_DIMS_NUM];
  int64_t strides_[MAX_DIMS_NUM];
};



struct OffsetCalculator_bianry {
  static constexpr int MAX_DIMS_NUM = 25;
  
  OffsetCalculator_bianry(int ndim, 
                          const int64_t *dims, 
                          const int64_t *strides,
                          const int64_t *minor_to_major)
      : ndim(ndim) {
    for (int i = 0; i < MAX_DIMS_NUM; i++) {
      dims_[i] = (i < ndim) ? dims[i] : 1;
      strides_[i] = (i < ndim) ? strides[i] : 0;
      minor_to_major_[i] = (i < ndim) ? minor_to_major[i] : 0;
    }
  }

  __device__ int idx2offset(int idx) const {
    int64_t offset = 0;
    int64_t location_i = 0;
#pragma unroll
    for (int64_t i = ndim - 1; i >= 0 ; i--) {
      // int64_t ii = minor_to_major_[i];
      int64_t ii = i;
      location_i = idx % dims_[ii];
      idx /= dims_[ii];
      offset += location_i * strides_[ii];
    }
    return offset;
  }

  int ndim;
  int64_t dims_[MAX_DIMS_NUM];
  int64_t strides_[MAX_DIMS_NUM];
  int64_t minor_to_major_[MAX_DIMS_NUM];
};