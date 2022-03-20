// This file is based on c10\core\CopyBytes.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's slightly modified for HICE's usage. 

#pragma once

#include "hice/core/device.h"
#include "hice/core/macros.h"

namespace hice {

using CopyBytesFunction = void (*)(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device);

struct _CopyBytesFunctionRegisterer {
  _CopyBytesFunctionRegisterer(
      DeviceType from,
      DeviceType to,
      CopyBytesFunction func_sync,
      CopyBytesFunction func_async = nullptr);
};

#define HICE_REGISTER_COPY_BYTES_FUNCTION(from, to, ...)           \
  namespace {                                                      \
  static _CopyBytesFunctionRegisterer HICE_ANONYMOUS_VARIABLE(     \
      g_copy_function)(from, to, __VA_ARGS__);                     \
  }

void copy_bytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async = false);

} // namespace hice
