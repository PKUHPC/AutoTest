cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aitisa_api VERSION 0.4.0 LANGUAGES C CXX)

##############################################
# General options
##############################################
option(AITISA_API_BUILD_SHARED_LIBS "Compiled as a shared library" ON)
option(AITISA_API_BUILD_TESTING "Build with testing enabled" ON)
option(AITISA_API_BUILD_AUTO_TESTING "Compile auto testing framework as a shared library" ON)
option(AITISA_API_BUILD_DOCS "Generate documents with Doxygen" ON)
option(AITISA_API_USE_GTEST_LOCAL "Download and build with GTest" ON)
option(AITISA_API_GENERATE_FIGURE "generate figure for test result" ON)
option(AITISA_API_PYTORCH "test with pytorch" ON)

if(AITISA_API_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
endif()

# Show the detail of compile messages
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_COLOR_MAKEFILE   ON)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
##############################################
# General setup
##############################################
# https://blog.kitware.com/cmake-and-the-default-build-type
#set(default_build_type "Release")
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(AITISA_API_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AITISA_API_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(AITISA_API_EXTERNAL_DIR ${AITISA_API_ROOT_DIR}/third_party)


include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
add_subdirectory(hice-master)

##############################################
# Create target and set properties
##############################################

include(cmake/public/Utils.cmake)

# Find all the files under `src/`
FILE(GLOB_RECURSE aitisa_api_srcs "src/*.c" "src/*.h")

# Create library target with all the source codes.
if(BUILD_SHARED_LIBS)
  add_library(aitisa_api SHARED ${aitisa_api_srcs})
  set_target_properties(aitisa_api
    PROPERTIES
      INTERFACE_POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME "aitisa_api"
      SOVERSION ${PROJECT_VERSION_MAJOR}
      C_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_definitions(aitisa_api
    PRIVATE
      AITISA_API_SHARED_LIBS
      AITISA_API_SHARED_LIBS_EXPORTS
  )
else()
  add_library(aitisa_api STATIC ${aitisa_api_srcs} hice_aitisa_test.cpp)
endif()

target_compile_features(aitisa_api PUBLIC c_std_99)
target_include_directories(aitisa_api
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

aitisa_api_make_interface_library(aitisa_api aitisa_api_interface)
add_library(aitisa_api::aitisa_api ALIAS aitisa_api_interface)

##############################################
# Testing
##############################################
if(AITISA_API_BUILD_TESTING)
  enable_testing()
  if(AITISA_API_USE_GTEST_LOCAL)
    include(cmake/local/GTest.cmake)
  else()
    include(cmake/public/GTest.cmake)
  endif()
  add_subdirectory(test)
endif()

##############################################
# Auto Testing
##############################################
if(AITISA_API_BUILD_AUTO_TESTING)
  if(NOT AITISA_API_BUILD_TESTING)
    enable_testing()
    if(AITISA_API_USE_GTEST_LOCAL)
      include(cmake/local/GTest.cmake)
    else()
      include(cmake/public/GTest.cmake)
    endif()
  endif()
  include(cmake/local/LibConfig.cmake)
  set(libconfig_build_include_dirs
          "${libconfig_SOURCE_DIR}/lib"
          "${libconfig_SOURCE_DIR}")
  include_directories(${libconfig_build_include_dirs})

  if(AITISA_API_PYTORCH)
    include(cmake/local/LibTorch.cmake)
    link_directories(/opt/intel/oneapi/mkl/latest/lib/intel64)
  endif()

  FILE(GLOB_RECURSE auto_test_srcs "auto_test/*.cpp" "auto_test/*.h")
  add_library(auto_test SHARED ${auto_test_srcs})
  set_target_properties(auto_test
    PROPERTIES
      INTERFACE_POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME "auto_test"
      SOVERSION ${PROJECT_VERSION_MAJOR}
      C_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_definitions(auto_test
    PRIVATE
      AUTO_TEST_SHARED_LIBS
      AUTO_TEST_SHARED_LIBS_EXPORTS
  )

  if(AITISA_API_GENERATE_FIGURE)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(auto_test Python3::Python Python3::Module)
    ADD_DEFINITIONS(-DAITISA_API_GENERATE_FIGURE)
  endif()

  if(AITISA_API_PYTORCH)
      set(Torch_DIR "${AITISA_API_EXTERNAL_DIR}/libtorch/libtorch-src/libtorch/share/cmake/Torch")
      find_package(Torch REQUIRED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

      target_link_libraries(auto_test "${TORCH_LIBRARIES}")
      ADD_DEFINITIONS(-DAITISA_API_PYTORCH)
#      include(cmake/local/TorchVision.cmake)
#      set(TorchVision_DIR "${AITISA_API_EXTERNAL_DIR}/torchvision/")

#      find_package(TorchVision REQUIRED)
#       include_directories(/home/xgy/Desktop/AutoTest/third_party/torchvision/torchvision-src/vision-0.13.1/torchvision/csrc)
#      target_link_libraries(auto_test /home/xgy/Desktop/AutoTest/third_party/torchvision/torchvision-build/libtorchvision.so)
#      target_link_libraries(auto_test PUBLIC TorchVision::TorchVision)
  endif()

  target_link_libraries(auto_test aitisa_api::aitisa_api aitisa_api::gmock aitisa_api::libconfig)

  add_library(aitisa_api::auto_test ALIAS auto_test)

  #add user lib
  target_link_libraries(auto_test hice::hice)

  #add user test code
  add_executable(hice_aitisa_test hice_aitisa_test.cpp)

  #link auto_test
  target_link_libraries(hice_aitisa_test auto_test)
  include_directories(hice-master)

endif()

##############################################
# Generate Documents
##############################################
if(AITISA_API_BUILD_DOCS)
  set(Doxygen_dir "/home/amax107/hice/ayl/doxygen-1.8.16")
  include(cmake/public/Doxygen.cmake)
endif()