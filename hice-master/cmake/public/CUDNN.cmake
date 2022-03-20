if(hice_public_cudnn_cmake_included)
  return()
endif()
set(hice_public_cudnn_cmake_included true)

# Find cuDNN 
if(DEFINED ENV{CUDNN_ROOT_DIR})
  set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder contains NVIDIA cuDNN")
else()
  set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
endif()

if(HICE_STATIC_LINK_CUDA)
 set(CUDNN_STATIC_LINKAGE ON)
else()
 set(CUDNN_STATIC_LINKAGE OFF)
endif()

find_package(CUDNN MODULE REQUIRED)

add_library(hice::cudnn INTERFACE IMPORTED)
set_property(
  TARGET hice::cudnn PROPERTY INTERFACE_LINK_LIBRARIES 
  ${CUDNN_LIBRARIES})
set_property(
  TARGET hice::cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CUDNN_INCLUDE_DIRS})

#add_library(hice::cudnn UNKNOWN IMPORTED)
#set_property(
#    TARGET hice::cudnn PROPERTY IMPORTED_LOCATION
#    ${CUDNN_LIBRARIES})
#set_property(
#    TARGET hice::cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
#    ${CUDNN_INCLUDE_DIR})
