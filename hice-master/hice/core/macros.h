#pragma once

#include "hice/core/export.h"

//// Compiler attributes
//#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
//// Compiler supports GCC-style attributes
//#define HICE_ATTRIBUTE_NORETURN __attribute__((noreturn))
//#define HICE_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
//#define HICE_ATTRIBUTE_NOINLINE __attribute__((noinline))
//#define HICE_ATTRIBUTE_UNUSED __attribute__((unused))
//#define HICE_ATTRIBUTE_COLD __attribute__((cold))
//#define HICE_ATTRIBUTE_WEAK __attribute__((weak))
//#define HICE_PACKED __attribute__((packed))
//#define HICE_MUST_USE_RESULT __attribute__((warn_unused_result))
//#define HICE_PRINHICE_ATTRIBUTE(string_index, first_to_check) \
//  __attribute__((__format__(__printf__, string_index, first_to_check)))
//#define HICE_SCANF_ATTRIBUTE(string_index, first_to_check) \
//  __attribute__((__format__(__scanf__, string_index, first_to_check)))
//#elif defined(_MSC_VER)
//// Non-GCC equivalents
//#define HICE_ATTRIBUTE_NORETURN __declspec(noreturn)
//#define HICE_ATTRIBUTE_ALWAYS_INLINE __forceinline
//#define HICE_ATTRIBUTE_NOINLINE
//#define HICE_ATTRIBUTE_UNUSED
//#define HICE_ATTRIBUTE_COLD
//#define HICE_ATTRIBUTE_WEAK
//#define HICE_MUST_USE_RESULT
//#define HICE_PACKED
//#define HICE_PRINHICE_ATTRIBUTE(string_index, first_to_check)
//#define HICE_SCANF_ATTRIBUTE(string_index, first_to_check)
//#else
//// Non-GCC equivalents
//#define HICE_ATTRIBUTE_NORETURN
//#define HICE_ATTRIBUTE_ALWAYS_INLINE
//#define HICE_ATTRIBUTE_NOINLINE
//#define HICE_ATTRIBUTE_UNUSED
//#define HICE_ATTRIBUTE_COLD
//#define HICE_ATTRIBUTE_WEAK
//#define HICE_MUST_USE_RESULT
//#define HICE_PACKED
//#define HICE_PRINHICE_ATTRIBUTE(string_index, first_to_check)
//#define HICE_SCANF_ATTRIBUTE(string_index, first_to_check)
//#endif

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define HICE_DISABLE_COPY_AND_ASSIGN(classname)   \
  classname(const classname&) = delete;           \
  classname& operator=(const classname&) = delete

#define HICE_CONCAT_IMPL(x, y) x##y
#define HICE_MACRO_CONCAT(x, y) HICE_CONCAT_IMPL(x, y)
#define HICE_MACRO_EXPAND(args) args

#ifdef __COUNTER__
#define HICE_ANONYMOUS_VARIABLE(str) HICE_MACRO_CONCAT(str, __COUNTER__)
#else
#define HICE_ANONYMOUS_VARIABLE(str) HICE_MACRO_CONCAT(str, __LINE__)
#endif