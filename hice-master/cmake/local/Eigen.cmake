if(hice_local_eigen_cmake_included)
  return()
endif()
set(hice_local_eigen_cmake_included true)

# For now we use the Netlib BLAS. However, Netlib LAPACK and BLAS 
# in the same project, So just include LAPACK cmake. In the future, 
# we may use other BLAS implementations and download them directly 
#include(cmake/local/LAPACK.cmake)
#add_library(hice::blas ALIAS cblas)
#return()

set(_external_target_name eigen)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(cmake/local/DownloadProject.cmake)
download_project(PROJ                ${_external_target_name} 
                 GIT_REPOSITORY      https://github.com/eigenteam/eigen-git-mirror.git
                 GIT_TAG             3.3.7 
                 GIT_PROGRESS        TRUE
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
                 PREFIX "${HICE_EXTERNAL_DIR}/${_external_target_name}"
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

#if(NOT TARGET eigen)
#  add_subdirectory(
#    ${${_external_target_name}_SOURCE_DIR} 
#    ${${_external_target_name}_BINARY_DIR})
#endif()
#add_library(hice::eigen ALIAS eigen)

if(NOT TARGET hice::eigen)
  add_library(hice::eigen INTERFACE IMPORTED)
  set_property(
    TARGET hice::eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${${_external_target_name}_SOURCE_DIR}")
endif()

unset(_external_target_name)