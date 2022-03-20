if(hice_public_gtest_cmake_included)
  return()
endif()
set(hice_public_gtest_cmake_included true)

hice_update_option(GTEST_ROOT ${GTEST_ROOT_DIR})
find_package(GTEST REQUIRED)
add_library(hice::gtest INTERFACE IMPORTED)
set_property(
  TARGET hice::gtest PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${GTEST_INCLUDE_DIRS})
set_property(
  TARGET hice::gtest PROPERTY INTERFACE_LINK_LIBRARIES
  ${GTEST_BOTH_LIBRARIES})