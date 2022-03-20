if(hice_public_mkl_cmake_included)
  return()
endif()
set(hice_public_mkl_cmake_included true)

find_package(MKL REQUIRED)

add_library(hice::mkl INTERFACE IMPORTED)
set_property(
  TARGET hice::mkl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKL_INCLUDE_DIR})
set_property(
  TARGET hice::mkl PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKL_LIBRARIES})