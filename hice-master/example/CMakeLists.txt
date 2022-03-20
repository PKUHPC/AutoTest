cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(example LANGUAGES CXX C)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -fPIC -pthread ${CMAKE_CXX_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CMakePrintHelpers)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# build hice
add_subdirectory(${CMAKE_SOURCE_DIR}/../ ${CMAKE_BINARY_DIR}/hice)
set(HICE_ROOT_DIR ${CMAKE_SOURCE_DIR}/..)
list(APPEND CMAKE_MODULE_PATH ${HICE_ROOT_DIR}/cmake/module)

# Use the path of your HICE location instead
# set(HICE_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../install_test")
# set(HICE_DIR "${HICE_INSTALL_DIR}/lib/cmake/hice-0.5.0")
# find_package(HICE REQUIRED)

add_executable(matmul matmul.cpp)
add_executable(plan_test plan_test.cpp)
add_executable(perf_test perf_test.cpp)

target_link_libraries(matmul PUBLIC hice::hice)
target_link_libraries(plan_test PUBLIC hice::hice)
target_link_libraries(perf_test PUBLIC hice::hice)

set_target_properties(matmul PROPERTIES 
  LINK_DIRECTORIES "${CMAKE_INSTALL_RPATH}"
)
set_target_properties(plan_test PROPERTIES 
  LINK_DIRECTORIES "${CMAKE_INSTALL_RPATH}"
)
set_target_properties(perf_test PROPERTIES 
  LINK_DIRECTORIES "${CMAKE_INSTALL_RPATH}"
)