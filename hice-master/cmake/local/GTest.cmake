if(hice_local_gtest_cmake_included)
  return()
endif()
set(hice_lcoal_gtest_cmake_included true)

set(gtest_SOURCE_DIR "${HICE_EXTERNAL_DIR}/gtest")
set(gtest_BINARY_DIR "${PROJECT_BINARY_DIR}/gtest")

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
if(NOT TARGET gtest_main)
  add_subdirectory(
    ${gtest_SOURCE_DIR}
    ${gtest_BINARY_DIR}
    EXCLUDE_FROM_ALL)
endif()

add_library(hice::gtest ALIAS gtest_main)
add_library(hice::gmock ALIAS gmock_main)