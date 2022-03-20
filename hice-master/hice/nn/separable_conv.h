#pragma once 

#include "hice/util/types.h"
#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"

namespace hice {

HICE_API Tensor &separable_conv_fwd(const Tensor &input, const Tensor &depth_kernel,
                                    const Tensor &point_kernel,  
                                    ConstIntArrayRef padding, ConstIntArrayRef stride, 
                                    Tensor &output);


} // namespace hice
