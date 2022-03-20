#include <cstring>

#include "hice/util/loguru.h"
#include "hice/util/copy_bytes.h"
#include "hice/device/cpu/context_cpu.h"

namespace hice {

namespace {
void copy_bytes_sync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
  if (nbytes == 0) {
    return;
  }
  HICE_CHECK_NOTNULL(src);
  HICE_CHECK_NOTNULL(dst);
  memcpy(dst, src, nbytes);
}
} // namespace

HICE_REGISTER_CONTEXT(DeviceType::CPU, CPUContext);

HICE_REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::CPU,
    copy_bytes_sync);

} // namespace hice