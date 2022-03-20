if(hice_public_cuda_cmake_included)
  return()
endif()
set(hice_public_cuda_cmake_included true)

# This file is used for CUDA configuration
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "HICE: Cannot find CUDA")
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

#get_filename_component(CUDA_TOOLKIT_ROOT_DIR
#  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#  DIRECTORY
#)

# we dont want to statically link cudart, because we rely on it's dynamic linkage in
# python (follow along torch/cuda/__init__.py and usage of cudaGetErrorName).
# Technically, we can link cudart here statically, and link libtorch_python.so
# to a dynamic libcudart.so, but that's just wasteful
SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

# Find CUDA
if(NOT CUDA_FOUND)
  find_package(CUDA QUIET REQUIRED)
endif()
message(STATUS "HICE: CUDA detected: " ${CUDA_VERSION})
message(STATUS "HICE: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "HICE: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})

# CUDA 9.0 & 9.1 require GCC version <= 5
# Although they support GCC 6, but a bug that wasn't fixed until 9.2 prevents
# them from compiling the std::tuple header of GCC 6.
# See Sec. 2.2.1 of
# https://developer.download.nvidia.com/compute/cuda/9.2/Prod/docs/sidebar/CUDA_Toolkit_Release_Notes.pdf
if ((CUDA_VERSION VERSION_EQUAL 9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  #if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
  #    NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
  #    CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
  #  message(FATAL_ERROR
  #    "CUDA ${CUDA_VERSION} is not compatible with std::tuple from GCC version "
  #    ">= 6. Please upgrade to CUDA 9.2 or use the following option to use "
  #    "another version (for example): \n"
  #    "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
  #endif()
elseif (CUDA_VERSION VERSION_EQUAL 8.0)
  # CUDA 8.0 requires GCC version <= 5
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
      NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
      CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "CUDA 8.0 is not compatible with GCC version >= 6. "
      "Use the following option to use another version (for example): \n"
      "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
  endif()
endif()

if(CUDA_FOUND)
  # Sometimes, we may mismatch nvcc with the CUDA headers we are
  # compiling with, e.g., if a ccache nvcc is fed to us by CUDA_NVCC_EXECUTABLE
  # but the PATH is not consistent with CUDA_HOME.  It's better safe
  # than sorry: make sure everything is consistent.
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_version.cc")
  file(WRITE ${file} ""
    "#include <cuda.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "  printf(\"%d.%d\", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);\n"
    "  return 0;\n"
    "}\n"
    )
  try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
    LINK_LIBRARIES ${CUDA_LIBRARIES}
    RUN_OUTPUT_VARIABLE cuda_version_from_header
    COMPILE_OUTPUT_VARIABLE output_var
  )
  if(NOT compile_result)
    message(FATAL_ERROR "HICE: Couldn't determine CUDA version from header: " ${output_var})
  endif()
  message(STATUS "HICE: CUDA header version is: " ${cuda_version_from_header})
  if(NOT ${cuda_version_from_header} STREQUAL ${CUDA_VERSION_STRING})
    # Force CUDA to be processed for again next time
    # TODO: I'm not sure if this counts as an implementation detail of
    # FindCUDA
    set(${cuda_version_from_findcuda} ${CUDA_VERSION_STRING})
    unset(CUDA_TOOLKIT_ROOT_DIR_INTERNAL CACHE)
    # Not strictly necessary, but for good luck.
    unset(CUDA_VERSION CACHE)
    # Error out
    message(FATAL_ERROR "FindCUDA says CUDA version is ${cuda_version_from_findcuda} (usually determined by nvcc), "
      "but the CUDA headers say the version is ${cuda_version_from_header}.  This often occurs "
      "when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to "
      "non-standard locations, without also setting PATH to point to the correct nvcc.  "
      "Perhaps, try re-running this command again with PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$PATH.  "
      "See above log messages for more diagnostics, and see https://github.com/pytorch/pytorch/issues/8092 for more details.")
  endif()
endif()

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if HICE_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static libraries installed.

# cudart. CUDA_LIBRARIES is actually a list, so we will make an interface
# library.
add_library(hice::cudart INTERFACE IMPORTED)
if(HICE_STATIC_LINK_CUDA)
    set_property(
        TARGET hice::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a" rt dl)
else()
    set_property(
        TARGET hice::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_LIBRARIES})
endif()
set_property(
    TARGET hice::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# thrust
# Head-only library
add_library(hice::thrust INTERFACE IMPORTED)
set_property(
  TARGET hice::thrust PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CUDA_INCLUDE_DIRS})

# cublas
# CUDA_CUBLAS_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(hice::cublas INTERFACE IMPORTED)
if(HICE_STATIC_LINK_CUDA)
  set_property(
    TARGET hice::cublas PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
else()
  set_property(
    TARGET hice::cublas PROPERTY INTERFACE_LINK_LIBRARIES
    ${CUDA_CUBLAS_LIBRARIES})
endif()
set_property(
  TARGET hice::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CUDA_INCLUDE_DIRS})

# cuSPARSE
add_library(hice::cusparse INTERFACE IMPORTED)
if(HICE_STATIC_LINK_CUDA)
  set_property(
    TARGET hice::cusparse PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a")
else()
  set_property(
    TARGET hice::cusparse PROPERTY INTERFACE_LINK_LIBRARIES
    ${CUDA_cusparse_LIBRARY})
endif()
set_property(
  TARGET hice::cusparse PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CUDA_INCLUDE_DIRS})

# curand
add_library(hice::curand INTERFACE IMPORTED)
if(HICE_STATIC_LINK_CUDA)
  set_property(
    TARGET hice::curand PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a")
else()
  set_property(
    TARGET hice::curand PROPERTY INTERFACE_LINK_LIBRARIES
    ${CUDA_curand_LIBRARY})
endif()
set_property(
  TARGET hice::curand PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CUDA_INCLUDE_DIRS})

# Setting nvcc arch flags
# The following statements is OK for compiling but not OK for running,
# but I don't know why. So let cmake automatically selects the underlying arch
#CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_LIST Auto)
#list(GET CUDA_NVCC_ARCH_LIST -2 -1 CURRENT_CUDA_NVCC_ARCH)
#list(APPEND CUDA_NVCC_FLAGS ${CURRENT_CUDA_NVCC_ARCH})

# Disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS_BLACKLIST "-Werror")

# Debug and Release symbol support
if (MSVC)
  if ((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo") OR (${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel"))
    if (${HICE_USE_MSVC_STATIC_RUNTIME})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MT")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MD")
    endif()
  elseif(${HICE_BUILD_TYPE} MATCHES "Debug")
    if (${HICE_USE_MSVC_STATIC_RUNTIME})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MTd")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MDd")
    endif()
  else()
    message(FATAL_ERROR "HICE: Unknown cmake build type " ${CMAKE_BUILD_TYPE})
  endif()
elseif (CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

STRING(REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

list(APPEND CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})

message(STATUS "HICE: CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")
