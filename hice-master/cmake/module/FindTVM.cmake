# - Try to find TVM
#
# The following variables are optionally searched for defaults
#  TVM_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  TVM_FOUND            : set to true if tvm is found.
#  TVM_INCLUDE_DIRS      : path to tvm include dir.
#  TVM_LIBRARIES_DIRS    : path to tvm libs dir.
#  TVM_LIBRARIES        : libtvm.so

IF (NOT TVM_FOUND)
  set(TVM_INCLUDE_DIRS)
  set(TVM_LIBRARIES_DIRS)
  set(TVM_LIBRARIES "libtvm.so")

  if (TVM_ROOT_DIR) 
    set(TVM_HOME ${TVM_ROOT_DIR})
  else()
    set(TVM_HOME $ENV{TVM_HOME})
  endif()

  find_path(TVM_LIBRARIES_DIRS ${TVM_LIBRARIES}
    HINTS ${TVM_HOME}
    PATH_SUFFIXES  build lib lib64)
    
  if(TVM_LIBRARIES_DIRS)
    set(TVM_INCLUDE_DIRS ${TVM_INCLUDE_DIRS} 
                        "${TVM_HOME}/include" 
                        "${TVM_HOME}/3rdparty/dmlc-core/include" 
                        "${TVM_HOME}/3rdparty/dlpack/include")
    # message(STATUS
    # ${TVM_INCLUDE_DIRS}
    # )
    set(TVM_FOUND TRUE)
  endif()
  
  # Standard termination
  if(NOT TVM_FIND_QUIETLY)
   if(TVM_FOUND)
     message(STATUS "A library with TVM API found.")
   else()
     if(TVM_FIND_REQUIRED)
       message(FATAL_ERROR
       "A required library with TVM API not found. Please specify library location."
       )
     else()
       message(STATUS
       "A library with TVM API not found. Please specify library location."
       )
     endif()
   endif()
  endif()

# Do nothing if TVM_FOUND was set before!
ENDIF (NOT TVM_FOUND)