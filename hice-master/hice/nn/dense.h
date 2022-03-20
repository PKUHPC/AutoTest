#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/scalar.h"
#include "hice/core/tensor.h"

namespace hice {

// input : 2-D with shape [batch, in_dim]

// weight : 2-D with shape [out_dim, in_dim]

// bias : optional, 1-D with shape [out_dim]

// output : 2-D with shape [batch, out_dim]

HICE_API Tensor& dense_fwd(const Tensor &input, const Tensor &weight, 
        hice::optional<Tensor> bias, Tensor &output);

HICE_API std::tuple<Tensor &, Tensor &> dense_bwd(const Tensor &input, const Tensor &weight, const Tensor &grad_output, 
              hice::optional<Tensor> bias, Tensor &grad_input, Tensor &grad_weight);

} // namespace hice
