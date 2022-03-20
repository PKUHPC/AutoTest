#include "hice/core/impl/device_guard_impl_interface.h"

namespace hice {
namespace impl {

std::atomic<const DeviceGuardImplInterface*>
device_guard_impl_registry[static_cast<size_t>(DeviceType::NumDeviceTypes)];

DeviceGuardImplRegister::DeviceGuardImplRegister(DeviceType type, const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

}
} // namespace hice::device
