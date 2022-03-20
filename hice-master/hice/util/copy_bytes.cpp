#include "hice/util/loguru.h"
#include "hice/util/copy_bytes.h"

namespace hice {

// First dimension of the array is `bool async`: 0 is sync,
// 1 is async (non-blocking)
static CopyBytesFunction g_copy_bytes[2][kNumDeviceTypes][kNumDeviceTypes];

_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,
    DeviceType toType,
    CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(fromType);
  auto to = static_cast<int>(toType);
  if (!func_async) {
    // default to the sync function
    func_async = func_sync;
  }
  HICE_CHECK(
      g_copy_bytes[0][from][to] == nullptr &&
      g_copy_bytes[1][from][to] == nullptr)
      << "Duplicate registration for device type pair "
      << fromType << ", " << toType;
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}

void copy_bytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async) {
  auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                         [static_cast<int>(dst_device.type())];

  HICE_CHECK_NOTNULL(ptr) << "No function found for copying from "
                     << src_device.type()
                     << " to "
                     << dst_device.type();

  ptr(nbytes, src, src_device, dst, dst_device);
}

}
