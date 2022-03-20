# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLAPACK
----------

Find LAPACK library

This module finds an installed fortran library that implements the
LAPACK linear-algebra interface (see http://www.netlib.org/lapack/).

The approach follows that taken for the autoconf macro file,
acx_lapack.m4 (distributed at
http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``BLAS_STATIC``
  if ``ON`` use static linkage

``BLAS_VENDOR``
  If set, checks only the specified vendor, if not set checks all the
  possibilities.  List of vendors valid in this module:

  * ``MKL``
  * ``OpenBLAS``
  * ``FLAME``
  * ``ACML``
  * ``Apple``
  * ``NAS``
  * ``Generic``

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LAPACK_FOUND``
  library implementing the LAPACK interface is found
``LAPACK_LINKER_FLAGS``
  uncached list of required linker flags (excluding -l and -L).
``LAPACK_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use LAPACK

.. note::

  C or CXX must be enabled to use Intel MKL

  For example, to use Intel MKL libraries and/or Intel compiler:

  .. code-block:: cmake

    set(BLAS_VENDOR MKL)
    find_package(LAPACK)
#]=======================================================================]

set(_lapack_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

# Check the language being used
if( NOT (CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED OR CMAKE_Fortran_COMPILER_LOADED) )
  if(LAPACK_FIND_REQUIRED)
    message(FATAL_ERROR "FindLAPACK requires Fortran, C, or C++ to be enabled.")
  else()
    message(STATUS "Looking for LAPACK... - NOT found (Unsupported languages)")
    return()
  endif()
endif()

if (CMAKE_Fortran_COMPILER_LOADED)
include(CheckFortranFunctionExists)
else ()
include(CheckFunctionExists)
endif ()
include(CMakePushCheckState)

cmake_push_check_state()
set(CMAKE_REQUIRED_QUIET ${LAPACK_FIND_QUIETLY})

set(LAPACK_FOUND FALSE)
set(LAPACK95_FOUND FALSE)

SET(LAPACK_ROOT_DIR "" CACHE STRING "LAPACK root directory")

# TODO: move this stuff to separate module

macro(Check_Lapack_Libraries LIBRARIES _prefix _name _flags _list _blas _threads)
# This macro checks for the existence of the combination of fortran libraries
# given by _list.  If the combination is found, this macro checks (using the
# Check_Fortran_Function_Exists macro) whether can link against that library
# combination using the name of a routine given by _name using the linker
# flags given by _flags.  If the combination of libraries is found and passes
# the link test, LIBRARIES is set to the list of complete library paths that
# have been found.  Otherwise, LIBRARIES is set to FALSE.

# N.B. _prefix is the prefix applied to the names of all cached variables that
# are generated internally and marked advanced by this macro.

set(_libraries_work TRUE)
set(${LIBRARIES})
set(_combined_name)
if (NOT _libdir)
  message("########FindLAPACK")
  if (WIN32)
    set(_libdir ENV LIB)
  elseif (APPLE)
    set(_libdir ENV DYLD_LIBRARY_PATH)
  else ()
    set(_libdir ENV LD_LIBRARY_PATH)
  endif ()
endif ()

list(APPEND _libdir "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")

foreach(_library ${_list})
  set(_combined_name ${_combined_name}_${_library})

  if(_libraries_work)
    if (BLAS_STATIC)
      if (WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif ()
      if (APPLE)
        set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
      else ()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif ()
    else ()
      if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # for ubuntu's libblas3gf and liblapack3gf packages
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .so.3gf)
      endif ()
    endif ()
    find_library(${_prefix}_${_library}_LIBRARY
      NAMES ${_library}
      PATHS ${_libdir}
      )
    mark_as_advanced(${_prefix}_${_library}_LIBRARY)
    set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
    set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
  endif()
endforeach()

if(_libraries_work)
  # Test this combination of libraries.
  if(UNIX AND BLAS_STATIC)
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} "-Wl,--start-group" ${${LIBRARIES}} ${_blas} "-Wl,--end-group" ${_threads})
  else()
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}} ${_blas} ${_threads})
  endif()
#  message("DEBUG: CMAKE_REQUIRED_LIBRARIES = ${CMAKE_REQUIRED_LIBRARIES}")
  if (NOT CMAKE_Fortran_COMPILER_LOADED)
    check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
  else ()
    check_fortran_function_exists(${_name} ${_prefix}${_combined_name}_WORKS)
  endif ()
  set(CMAKE_REQUIRED_LIBRARIES)
  set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  #message("DEBUG: ${LIBRARIES} = ${${LIBRARIES}}")
endif()

 if(_libraries_work)
   set(${LIBRARIES} ${${LIBRARIES}} ${_blas} ${_threads})
 else()
    set(${LIBRARIES} FALSE)
 endif()

endmacro()


set(LAPACK_LINKER_FLAGS)
set(LAPACK_LIBRARIES)
set(LAPACK95_LIBRARIES)


if(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  find_package(BLAS)
else()
  find_package(BLAS REQUIRED)
endif()


if(BLAS_FOUND)
  set(LAPACK_LINKER_FLAGS ${BLAS_LINKER_FLAGS})
  if (NOT $ENV{BLAS_VENDOR} STREQUAL "")
    set(BLAS_VENDOR $ENV{BLAS_VENDOR})
  else ()
    if(NOT BLAS_VENDOR)
      set(BLAS_VENDOR "All")
    endif()
  endif ()

  #intel lapack
  if (BLAS_VENDOR STREQUAL "MKL" OR BLAS_VENDOR STREQUAL "All")
    IF(MKL_LAPACK_LIBRARIES)
      SET(LAPACK_LIBRARIES ${MKL_LAPACK_LIBRARIES} ${MKL_LIBRARIES})
    ELSE(MKL_LAPACK_LIBRARIES)
      SET(LAPACK_LIBRARIES ${MKL_LIBRARIES})
    ENDIF(MKL_LAPACK_LIBRARIES)
    SET(LAPACK_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    SET(LAPACK_INFO "mkl")
  ENDIF()
  
  if (BLAS_VENDOR STREQUAL "Goto" OR BLAS_VENDOR STREQUAL "All")
   if(NOT LAPACK_LIBRARIES)
    check_lapack_libraries(
    LAPACK_LIBRARIES
    LAPACK
    cheev
    ""
    "goto2"
    "${BLAS_LIBRARIES}"
    ""
    "${LAPACK_ROOT_DIR}"
    )
   endif()
  endif ()
  
  if (BLAS_VENDOR STREQUAL "OpenBLAS" OR BLAS_VENDOR STREQUAL "All")
   if(NOT LAPACK_LIBRARIES)
    check_lapack_libraries(
    LAPACK_LIBRARIES
    LAPACK
    cheev
    ""
    "openblas"
    "${BLAS_LIBRARIES}"
    ""
    "${LAPACK_ROOT_DIR}"
    )
   endif()
  endif ()
  
  if (BLAS_VENDOR STREQUAL "FLAME" OR BLAS_VENDOR STREQUAL "All")
   if(NOT LAPACK_LIBRARIES)
    check_lapack_libraries(
    LAPACK_LIBRARIES
    LAPACK
    cheev
    ""
    "flame"
    "${BLAS_LIBRARIES}"
    ""
    "${LAPACK_ROOT_DIR}"
    )
   endif()
  endif ()
  
  #acml lapack
   if (BLAS_VENDOR MATCHES "ACML" OR BLAS_VENDOR STREQUAL "All")
     if (BLAS_LIBRARIES MATCHES ".+acml.+")
       set (LAPACK_LIBRARIES ${BLAS_LIBRARIES})
     endif ()
   endif ()
  
  # Apple LAPACK library?
  if (BLAS_VENDOR STREQUAL "Apple" OR BLAS_VENDOR STREQUAL "All")
   if(NOT LAPACK_LIBRARIES)
    check_lapack_libraries(
    LAPACK_LIBRARIES
    LAPACK
    cheev
    ""
    "Accelerate"
    "${BLAS_LIBRARIES}"
    ""
    "${LAPACK_ROOT_DIR}"
    )
   endif()
  endif ()
  if (BLAS_VENDOR STREQUAL "NAS" OR BLAS_VENDOR STREQUAL "All")
    if ( NOT LAPACK_LIBRARIES )
      check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "vecLib"
      "${BLAS_LIBRARIES}"
      ""
      "${LAPACK_ROOT_DIR}"
      )
    endif ()
  endif ()
  # Generic LAPACK library?
  if (BLAS_VENDOR STREQUAL "Generic" OR
      BLAS_VENDOR STREQUAL "ATLAS" OR
      BLAS_VENDOR STREQUAL "All")
    if ( NOT LAPACK_LIBRARIES )
      check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "lapack"
      "${BLAS_LIBRARIES}"
      ""
      "${LAPACK_ROOT_DIR}"
      )
    endif ()
  endif ()

else()
  message(STATUS "LAPACK requires BLAS")
endif()

if(LAPACK_LIBRARIES)
 set(LAPACK_FOUND TRUE)
else()
 set(LAPACK_FOUND FALSE)
endif()

if(NOT LAPACK_FIND_QUIETLY)
 if(LAPACK_FOUND)
   message(STATUS "A library with LAPACK API found.")
 else()
   if(LAPACK_FIND_REQUIRED)
     message(FATAL_ERROR
     "A required library with LAPACK API not found. Please specify library location."
     )
   else()
     message(STATUS
     "A library with LAPACK API not found. Please specify library location."
     )
   endif()
 endif()
endif()

cmake_pop_check_state()
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_lapack_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})