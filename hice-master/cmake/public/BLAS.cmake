if(hice_public_blas_cmake_included)
  return()
endif()
set(hice_public_blas_cmake_included true)

find_package(BLAS REQUIRED)
add_library(hice::blas INTERFACE IMPORTED)
set_property(
  TARGET hice::blas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${BLAS_INCLUDE_DIRS})
set_property(
  TARGET hice::blas PROPERTY INTERFACE_LINK_LIBRARIES
  ${BLAS_LIBRARIES})
