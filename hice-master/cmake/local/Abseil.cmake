if(hice_local_absl_cmake_included)
  return()
endif()
set(hice_local_absl_cmake_included true)

set(absl_SOURCE_DIR "${HICE_EXTERNAL_DIR}/absl")
set(absl_BINARY_DIR_SUBPROJECT "${PROJECT_BINARY_DIR}/absl/as_subproject")
set(absl_BINARY_DIR_PROJECT "${PROJECT_BINARY_DIR}/absl/as_project")
set(absl_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(absl_DIR "${absl_INSTALL_DIR}/lib/cmake/absl" 
    Cache STRING "Make local absl_DIR available to HICECONFIG.cmake")

# Abseil's install rules ares disabled when added as subdirectory.
# There is dirty tricks to install Abseil along with hice.
# step 1. Build Abseil as a subproject, so dependency can be found in build time.
if(NOT TARGET absl)
  add_subdirectory(
    ${absl_SOURCE_DIR}
    ${absl_BINARY_DIR_SUBPROJECT}
  )
endif()

add_library(hice::absl INTERFACE IMPORTED)
set_property(
  TARGET 
    hice::absl 
  PROPERTY INTERFACE_LINK_LIBRARIES
    absl::base absl::algorithm absl::strings absl::optional
)


# step 2. Configure, Build and install Abseil independently at installing time.
install(CODE 
"
  # step 2-1. Configure Abseil into ${absl_BINARY_DIR_PROJECT} as a project.
  message(STATUS \"Abseil Configuring ...\")
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${absl_BINARY_DIR_PROJECT})
  execute_process(COMMAND 
      ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${absl_INSTALL_DIR}
      ${absl_SOURCE_DIR} WORKING_DIRECTORY ${absl_BINARY_DIR_PROJECT}
      RESULT_VARIABLE _res
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_QUIET
  )
  # message(STATUS ${_res})
  message(STATUS \"Abseil Configured.\")
  message(STATUS \"Abseil Building and installing ...\")
  # step 2-2. Build and install.
  execute_process(COMMAND 
      ${CMAKE_COMMAND} 
        --build ${absl_BINARY_DIR_PROJECT} 
        --target install 
        -j 16 
        OUTPUT_QUIET
  )
  message(STATUS \"Abseil built and installed.\")
")