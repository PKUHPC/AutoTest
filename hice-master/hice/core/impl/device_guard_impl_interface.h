// This file is based on c10\core\impl\DeviceGuardImplInterface.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 

#pragma once

#include <atomic>

#include "hice/core/macros.h"
#include "hice/util/loguru.h"
#include "hice/core/device.h"
#include "hice/core/stream.h"

namespace hice {
namespace impl {

/**
 * DeviceGuardImplInterface represents the virtual interface which provides
 * functionality to provide an RAII class for device and stream switching,
 * via DeviceGuard.  Every distinct device type, e.g., CUDA and HIP, is
 * expected to implement and register an implementation of this interface.
 * All classes which inherit from DeviceGuardImplInterface should be declared
 * 'final'.
 *
 * This class exists because we provide a unified interface for performing
 * device guards via DeviceGuard, but we cannot assume that we have actually
 * compiled against the, e.g., CUDA library, which actually implements
 * this guard functionality.  In this case, a dynamic dispatch is required
 * to cross the library boundary.
 *
 * If possible, you should directly use implementations of this interface;
 * those uses will be devirtualized.
 */
struct  DeviceGuardImplInterface {
  /**
   * Return the type of device managed by this guard implementation.
   */
  virtual DeviceType type() const = 0;

  /**
   * Set the current device to Device, and return the previous Device.
   */
  virtual Device exchange_device(Device) const = 0;
  // NB: Implementations of exchange_device can be a bit boilerplatey.  You might
  // consider replacing exchange_device with a non-virtual function with a baked
  // in implementation; however, note that this will triple the number of
  // virtual calls (when you implement exchange_device in a final subclass,
  // the compiler gets to devirtualize everything; it won't do that if you don't
  // define it in the subclass!)  A common way to solve this problem is to use
  // some sort of CRTP; however, we can template DeviceGuardImplInterface since
  // we really *do* need it to be virtual.  A little boilerplate seems easiest
  // to explain.  (Another way around this problem is to provide inline
  // functions that provide the default implementations, but this seems a little
  // hard to explain.  In any case, we're only going to have on order of ten
  // implementations of this anyway.)

  /**
   * Get the current device.
   */
  virtual Device get_device() const = 0;

  /**
   * Set the current device to Device.
   */
  virtual void set_device(Device) const = 0;

  /**
   * Set the current device to Device, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  virtual void unchecked_set_device(Device) const noexcept = 0;

  /**
   * Get the current stream for a given device.
   */
  virtual Stream get_stream(Device) const noexcept = 0;

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  virtual Stream exchange_stream(Stream) const noexcept = 0;

  /**
   * Get the number of devices.  WARNING: This is REQUIRED to not raise
   * an exception.  If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  virtual DeviceIndex device_count() const noexcept = 0;

  /**
   * Intended use of this class is to leak the DeviceGuardImpl at program end.
   * So you better not call the destructor, buster!
   */
  virtual ~DeviceGuardImplInterface() = default;
};

// The registry is NON-owning.  Each stored pointer is std::atomic so
// that under all interleavings of registry calls the structure is
// race-free.  This doesn't cost us anything on reads in X86.  (An
// unsynchronized implementation probably is OK too, but I didn't want
// to prove that we never read from device_guard_impl_registry at the
// same time some registration is occurring.  Shiver.)
//
// I'd like this registry to be valid even at program destruction time
// (in case someone uses a DeviceGuard in a destructor to do some cleanup
// in the CUDA API.)  Since there are no direct accesses of the underlying
// owning objects which I can use to enforce initialization order (unlike
// in a Meyer singleton), it implies that you must *leak* objects when
// putting them in the registry.  This is done by deleting the destructor
// on DeviceGuardImplInterface.
extern  std::atomic<const DeviceGuardImplInterface*>
device_guard_impl_registry[static_cast<size_t>(DeviceType::NumDeviceTypes)];

// I can't conveniently use hice/util/Registry.h for the following reason:
// hice/util/Registry.h gives me a slow way of Createing a object of some
// interface from the registry, but no way of quickly accessing an already
// created object.  I'll be banging on get_device_guard_impl every time we do a
// DeviceGuard, so I really don't want to be doing an unordered_map lookup.
// Better if the registration mechanism directly drops its implementation
// into device_guard_impl_registry.

class  DeviceGuardImplRegister {
public:
  DeviceGuardImplRegister(DeviceType, const DeviceGuardImplInterface*);
};

#define HICE_REGISTER_GUARD_IMPL(DevType, DeviceGuardImpl)               \
  static hice::impl::DeviceGuardImplRegister HICE_ANONYMOUS_VARIABLE( \
      g_##DeviceType)(hice::DeviceType::DevType, new DeviceGuardImpl());

inline const DeviceGuardImplInterface* get_device_guard_impl(DeviceType type) {
  auto p = device_guard_impl_registry[static_cast<size_t>(type)].load();
  HICE_CHECK(p) << "DeviceGuardImpl for " << type << " is not available";
  return p;
}

inline bool has_device_guard_impl(DeviceType type) {
  return device_guard_impl_registry[static_cast<size_t>(type)].load();
}

}} // namespace hice::device
