// This file is based on Eigen\src\Core\util\Memory.h from Eigen. 
// Eigen is Mozilla Public License v. 2.0  licensed, as found in its LICENSE file.
// From https://bitbucket.org/eigen/eigen
// And it's slightly modified for HICE's usage.

#pragma once

#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <limits>

// These implementations are taken from Eigen library

namespace hice {

static constexpr size_t kDefaultAlignBytes = 64;

namespace detail {

inline void* _aligned_malloc(std::size_t size, std::size_t align_bytes) {
  if (size == 0) return 0;
  void *unaligned = std::malloc(size + align_bytes);
	if (unaligned == 0) return 0;
	void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(unaligned) & 
				                                   ~(std::size_t(align_bytes - 1))) + align_bytes);
	*(reinterpret_cast<void**>(aligned) - 1) = unaligned;
  return aligned;
}

inline void _aligned_free(void *ptr) {
  if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
}

inline void* _aligned_realloc(void* ptr, std::size_t new_size, std::size_t align_bytes) {
  if (ptr == 0) return _aligned_malloc(new_size, kDefaultAlignBytes);
  void *pre_unaligned = *(reinterpret_cast<void**>(ptr) - 1);
	std::ptrdiff_t pre_offset = static_cast<char *>(ptr) - static_cast<char *>(pre_unaligned);
  void *new_unaligned = std::realloc(pre_unaligned, new_size + align_bytes);
  if (new_unaligned == 0) return 0;
  void *new_aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(new_unaligned) & 
				                                   ~(std::size_t(align_bytes-1))) + align_bytes);
	void *pre_aligned = static_cast<char *>(new_unaligned) + pre_offset;
	if(new_aligned != pre_aligned)
		std::memmove(new_aligned, pre_aligned, new_size);
  *(reinterpret_cast<void**>(new_aligned) - 1) = new_unaligned;
  return new_aligned;
}

} // namesapce detail

inline void* aligned_malloc(std::size_t size, 
                            std::size_t align_bytes = kDefaultAlignBytes) {
  return detail::_aligned_malloc(size, align_bytes);
}

inline void aligned_free(void *ptr) {
  detail::_aligned_free(ptr);
}

inline void* aligned_realloc(void* ptr, std::size_t new_size, 
                              std::size_t align_bytes = kDefaultAlignBytes) {
  return detail::_aligned_realloc(ptr, new_size, align_bytes);
}


} // namespace hice
