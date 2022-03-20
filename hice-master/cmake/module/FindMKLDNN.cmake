# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn

IF (NOT MKLDNN_FOUND)

  SET(MKLDNN_ROOT_DIR "" CACHE STRING "MKLDNN root directory")
  SET(MKLDNN_LIBRARIES)
  SET(MKLDNN_INCLUDE_DIR)
  
  FIND_PATH(MKLDNN_INCLUDE_DIR 
            NAMES mkldnn.hpp mkldnn.h 
            PATHS ${MKLDNN_ROOT_DIR} 
            PATH_SUFFIXES include)
  
  FIND_LIBRARY(MKLDNN_LIBRARIES
               NAMES mkldnn 
               PATHS  ${MKLDNN_ROOT_DIR} 
               PATH_SUFFIXES lib64 lib)

  
  set(_mkldnn_use_mkl TRUE)

  FIND_PACKAGE(MKL QUIET)
  IF(MKL_FOUND)
    # Append to mkldnn dependencies
    LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
    LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  ELSE(MKL_FOUND)
    IF(APPLE)
      SET(_mklml_inner_libs mklml iomp5)
    ELSE(APPLE)
      SET(_mklml_inner_libs mklml_intel iomp5)
    ENDIF(APPLE)

    FOREACH(_mklml_inner_lib ${_mklml_inner_libs})
      STRING(TOUPPER ${_mklml_inner_lib} _mklml_inner_lib_upper)
      FIND_LIBRARY(${_mklml_inner_lib_upper}_LIBRARY
            NAMES ${_mklml_inner_lib}
            PATHS  ${MKLDNN_ROOT_DIR}
            PATH_SUFFIXES lib lib64)
      MARK_AS_ADVANCED(${_mklml_inner_lib_upper}_LIBRARY)
      IF(NOT ${_mklml_inner_lib_upper}_LIBRARY)
        set(_mkldnn_use_mkl FALSE)
        break()
      ENDIF(NOT ${_mklml_inner_lib_upper}_LIBRARY)
      LIST(APPEND MKLDNN_LIBRARIES ${${_mklml_inner_lib_upper}_LIBRARY})
    ENDFOREACH(_mklml_inner_lib)
  ENDIF(MKL_FOUND)

  IF(MKLDNN_INCLUDE_DIR AND MKLDNN_LIBRARIES)
    set(MKLDNN_FOUND TRUE)
  else()
    set(MKLDNN_FOUND FALSE)
  endif()

  if(NOT MKLDNN_FIND_QUIETLY)
   if(MKLDNN_FOUND)
     if(_mkldnn_use_mkl)
       message(STATUS "A library with MKLDNN API found.")
     else()
       message(WARNING "A library with MKLDNN API found. But make sure it can"
                       "work without MKL support or add MKL support manually later."
       )
     endif()
   else()
     if(MKLDNN_FIND_REQUIRED)
       message(FATAL_ERROR
       "A required library with MKLDNN API not found. Please specify library location."
       )
     else()
       message(STATUS
       "A library with MKLDNN API not found. Please specify library location."
       )
     endif()
   endif()
  endif()

  unset(_mkldnn_use_mkl)

ENDIF(NOT MKLDNN_FOUND)
