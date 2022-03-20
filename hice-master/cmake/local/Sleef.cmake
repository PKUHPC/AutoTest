if(hice_local_sleef_cmake_included)
  return()
endif()
set(hice_local_sleef_cmake_included true)

set(sleef_SOURCE_DIR "${HICE_EXTERNAL_DIR}/sleef")
set(sleef_BINARY_DIR "${PROJECT_BINARY_DIR}/sleef")

# Preserve values for the main build
set(OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(OLD_BUILD_TESTS ${BUILD_TESTS})
set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS)
# Bump up optimization level for sleef to -O1, since at -O0 the compiler
# excessively spills intermediate vector registers to the stack
# and makes things run impossibly slowly
set(OLD_CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
IF(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
  string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
ELSE()
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
ENDIF()

# set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
set(OLD_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND
    CMAKE_C_COMPILER_VERSION VERSION_GREATER 6.9 AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
  set(GCC_7 True)
else()
  set(GCC_7 False)
endif()
if(GCC_7)
  set(CMAKE_BUILD_TYPE Release)  # Always build Sleef as a Release build to work around a gcc-7 bug
endif()

if(NOT TARGET sleef)
  add_subdirectory(
    ${sleef_SOURCE_DIR}
    ${sleef_BINARY_DIR}
  )
endif()

include_directories(${sleef_BINARY_DIR}/include)
link_directories(${PROJECT_BINARY_DIR}/lib)
# link_directories(${${_external_target_name}_BINARY_DIR}/lib)

add_library(hice::sleef INTERFACE IMPORTED)
set_property(
  TARGET hice::sleef
  PROPERTY INTERFACE_LINK_LIBRARIES sleef)

#### recover flags
set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
# Set these back. TODO: Use SLEEF_ to pass these instead
set(BUILD_SHARED_LIBS ${OLD_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
set(BUILD_TESTS ${OLD_BUILD_TESTS} CACHE BOOL "Build tests" FORCE)
if(GCC_7)
  set(CMAKE_BUILD_TYPE ${OLD_CMAKE_BUILD_TYPE})
endif()