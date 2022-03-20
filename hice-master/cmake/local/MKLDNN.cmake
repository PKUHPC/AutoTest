if(hice_local_mkldnn_cmake_included)
  return()
endif()
set(hice_local_mkldnn_camke_included true)

find_package(MKL QUIET)
if(NOT MKL_FOUND)
  if (NOT IS_DIRECTORY "${HICE_EXTERNAL_DIR}/mkldnn/external")
    if (UNIX)
      execute_process(
        COMMAND "${HICE_EXTERNAL_DIR}/mkldnn/scripts/prepare_mkl.sh"
        RESULT_VARIABLE _result)
    else ()
      execute_process(
        COMMAND "${HICE_EXTERNAL_DIR}/mkldnn/scripts/prepare_mkl.bat"
        RESULT_VARIABLE _result)
    endif()
  endif()
endif()

set(MKLROOT ${MKL_ROOT_DIR} CACHE STRING "" FORCE)
set(MKLDNN_THREADING "OMP:COMP" CACHE STRING "" FORCE)
set(MKLDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MKLDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

if(NOT TARGET mkldnn)
  add_subdirectory(
    ${HICE_EXTERNAL_DIR}/mkldnn
    ${PROJECT_BINARY_DIR}/mkldnn
  )
endif()

add_library(hice::mkldnn ALIAS mkldnn)
