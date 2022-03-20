if(hice_public_mkldnn_cmake_included)
  return()
endif()
set(hice_public_mkldnn_cmake_included true)

find_package(MKLDNN REQUIRED)
add_library(hice::mkldnn INTERFACE IMPORTED)
set_property(
  TARGET hice::mkldnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKLDNN_INCLUDE_DIRS})
set_property(
  TARGET hice::mkldnn PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKLDNN_LIBRARIES})
