#include "hice/core/context.h"

namespace hice {

HICE_DEFINE_TYPED_REGISTRY(
    ContextRegistry,
    DeviceType,
    DeviceContext,
    std::unique_ptr,
    Device);

} // namespace hice 
