if(hice_public_python_cmake_included)
  return()
endif()
set(hice_public_python_cmake_included true)

if(NOT PYTHON_INCLUDE_DIRS)
  find_package(Python3 3.6 REQUIRED)
  find_package(PythonLibs 3.6 REQUIRED)
endif()

add_library(hice::python INTERFACE IMPORTED)
set_property(
  TARGET hice::python PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${PYTHON_INCLUDE_DIRS})
set_property(
  TARGET hice::python PROPERTY INTERFACE_LINK_LIBRARIES
  ${PYTHON_LIBRARIES})