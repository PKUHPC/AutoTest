if(hice_public_eigen_cmake_included)
  return()
endif()
set(hice_public_eigen_cmake_included true)

find_package(Eigen3 REQUIRED NO_MODULE)
add_library(hice::eigen ALIAS Eigen3::Eigen)
