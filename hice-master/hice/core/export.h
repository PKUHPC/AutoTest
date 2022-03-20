#pragma once

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
  #define HICE_EXPORT __declspec(dllexport)
  #define HICE_IMPORT __declspec(dllimport)
  #define HICE_INNER
#else
  #if __GNUC__ >= 4
    #define HICE_EXPORT __attribute__ ((visibility ("default")))
    #define HICE_IMPORT __attribute__ ((visibility ("default")))
    #define HICE_INNER  __attribute__ ((visibility ("hidden")))
  #else
    #define HICE_EXPORT
    #define HICE_IMPORT
    #define HICE_INNER
  #endif
#endif

// Now we use the generic helper definitions above to define HICE_PUBLIC and HICE_LOCAL.
// HICE_PUBLIC is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// HICE_LOCAL is used for non-api symbols.

#ifdef HICE_SHARED_LIBS // defined if HICE is compiled as a shared library 
  #ifdef HICE_SHARED_LIBS_EXPORTS // defined if we are building the HICE shared library (instead of using it)
    #define HICE_API HICE_EXPORT
  #else
    #define HICE_API HICE_IMPORT
  #endif // HICE_SHARED_LIB_EXPORTS
  #define HICE_LOCAL HICE_INNER
#else // HICE_SHARED_LIB is not defined: this means HICE is a static lib.
  #define HICE_API
  #define HICE_LOCAL
#endif // HICE_DLL