#pragma once 
#include <tuple>
#include <string>

#include "hice/core/tensor.h"

namespace hice {

// Operators
HICE_API std::tuple<Tensor, Tensor, Tensor, Tensor> load_dataset(
    std::string dataset_name);

} // namesapce hice
