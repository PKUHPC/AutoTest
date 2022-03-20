if(hice_public_tvm_cmake_included)
  return()
endif()
set(hice_public_tvm_cmake_included true)

# set(hice_tvm_ops_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hice/tvm)
# set(hice_tvm_ops_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hice_tvm_ops)

# generate libhice_tvm_ops.so
# message(STATUS "")
# message(STATUS "")
# message(STATUS "")
# message(STATUS "Generating libhice_tvm_ops.so ...")
# set(HICE_TVM_OPS_OBJ_FULL "hice_tvm_ops.o")
# set(HICE_TVM_OPS_LIB_FULL "libhice_tvm_ops.so")
# set(HICE_TVM_OPS_LIB "hice_tvm_ops")
# execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${hice_tvm_ops_BINARY_DIR})
# execute_process(COMMAND 
#     ${CMAKE_COMMAND}
#     ${hice_tvm_ops_SOURCE_DIR} WORKING_DIRECTORY ${hice_tvm_ops_BINARY_DIR}
#     RESULT_VARIABLE _res
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     # OUTPUT_QUIET
# )
# execute_process(COMMAND 
#     ${CMAKE_COMMAND} --build ${hice_tvm_ops_BINARY_DIR}
#     RESULT_VARIABLE _res
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     # OUTPUT_QUIET
# )
# execute_process(COMMAND "./tvm_lib_generator" WORKING_DIRECTORY ${hice_tvm_ops_BINARY_DIR}
#     RESULT_VARIABLE _res
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     # OUTPUT_QUIET
# )
# message(STATUS "  Generated.")
# message(STATUS "")
# message(STATUS "")
# message(STATUS "")

find_package(TVM REQUIRED)
add_library(hice::tvm INTERFACE IMPORTED)

# set_property(
#   TARGET hice::tvm PROPERTY INTERFACE_INCLUDE_DIRECTORIES
#   ${TVM_INCLUDE_DIRS})
# set_property(
#   TARGET hice::tvm PROPERTY INTERFACE_LINK_DIRECTORIES
#   ${TVM_LIBRARIES_DIRS} ${hice_tvm_ops_BINARY_DIR})
# set_property(
#   TARGET hice::tvm PROPERTY INTERFACE_LINK_LIBRARIES
#   ${TVM_LIBRARIES} ${HICE_TVM_OPS_LIB})

set_property(
  TARGET hice::tvm PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${TVM_INCLUDE_DIRS})
set_property(
  TARGET hice::tvm PROPERTY INTERFACE_LINK_DIRECTORIES
  ${TVM_LIBRARIES_DIRS})
set_property(
  TARGET hice::tvm PROPERTY INTERFACE_LINK_LIBRARIES
  ${TVM_LIBRARIES} ${HICE_TVM_OPS_LIB})