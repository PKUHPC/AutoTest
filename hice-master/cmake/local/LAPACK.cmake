if(hice_local_lapack_cmake_included)
  return()
endif()
set(hice_local_lapack_cmake_included true)

set(_external_target_name lapack)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(cmake/local/DownloadProject.cmake)
download_project(PROJ                ${_external_target_name} 
                 GIT_REPOSITORY      https://github.com/Reference-LAPACK/lapack-release.git
                 GIT_TAG             d97a30482e005c90822c12a8ea684d5200a8e314
                 GIT_PROGRESS        TRUE
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
                 PREFIX "${HICE_EXTERNAL_DIR}/${_external_target_name}"
)

#set(USE_OPTIMIZED_BLAS ON CACHE BOOL "" FORCE)
#set(USE_OPTIMIZED_LAPACK ON CACHE BOOL "" FORCE)
set(CBLAS ON CACHE BOOL "" FORCE)
set(LAPACKE ON CACHE BOOL "" FORCE)

if(NOT TARGET lapacke)
  add_subdirectory(
    ${${_external_target_name}_SOURCE_DIR} 
    ${${_external_target_name}_BINARY_DIR})
endif()

add_library(hice::lapack ALIAS lapacke) 
include_directories(${${_external_target_name}_BINARY_DIR}/include)

unset(_external_target_name)