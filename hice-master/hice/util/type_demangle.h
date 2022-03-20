// This file is based on c10\util\Type.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage.

#pragma once

#include <cstddef>
#include <string>
#include <typeinfo>

namespace hice {

/// Utility to demangle a C++ symbol name.
std::string demangle(const char* name);

/// Returns the printable name of the type.
template <typename T>
inline const char* demangle_type() {
#ifdef __GXX_RTTI
  static const auto& name = *(new std::string(demangle(typeid(T).name())));
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

} // namespace hice