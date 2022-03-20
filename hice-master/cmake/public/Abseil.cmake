if(hice_public_abseil_cmake_included)
  return()
endif()
set(hice_public_abseil_cmake_included true)

# set(absl_INSTALL_ROOT "~/hice/likesen/absl/install")
# set(absl_DIR "${absl_INSTALL_ROOT}/lib/cmake/absl")

find_package(absl REQUIRED)
add_library(hice::absl INTERFACE IMPORTED)
set_property(
  TARGET 
    hice::absl 
  PROPERTY INTERFACE_LINK_LIBRARIES
    absl::base absl::algorithm absl::strings absl::optional
)