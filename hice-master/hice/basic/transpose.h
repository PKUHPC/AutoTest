#pragma once

#include "hice/core/tensor.h"
#include "hice/core/scalar.h"
#include "hice/core/dispatch.h"

namespace hice {

/// NOTE: transpose and transpose_matrix does NOT change the order of underlying data,
/// it just change the layout. If you want to change the order of underlying data,
/// call contiguous after transpose

// conjugate is not supportted.
HICE_API Tensor transpose(const Tensor& tensor, ConstIntArrayRef perm = {}, bool conjugate = false);

// do transpose only on the two inner-most dims
HICE_API Tensor transpose_matrix(const Tensor& tensor, bool conjugate = false);

HICE_API Tensor& transpose_(Tensor& tensor, ConstIntArrayRef perm = {}, bool conjugate = false);

HICE_API Tensor& transpose_matrix_(Tensor& tensor, bool conjugate = false);


// The following version of transpose is implemented by Ye Zilingfeng.
#if 0
using transpose_kernel_fn_type = void (*)(const Tensor& input, 
                                          ConstIntArrayRef perm_dims,
                                          Tensor & output);

HICE_DECLARE_DISPATCHER(transpose_dispatcher , transpose_kernel_fn_type);
#endif

} // namespace hice
