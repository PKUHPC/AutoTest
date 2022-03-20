if(hice_public_lapack_cmake_included)
  return()
endif()
set(hice_public_lapack_cmake_included true)

find_package(LAPACK REQUIRED)
add_library(hice::lapack INTERFACE IMPORTED)
set_property(
  TARGET hice::lapack PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${LAPACK_INCLUDE_DIRS})
set_property(
  TARGET hice::lapack PROPERTY INTERFACE_LINK_LIBRARIES
  ${LAPACK_LIBRARIES})
