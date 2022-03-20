set(module_name core)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/index_util_test.cpp
  ${module_name}/layout_test.cpp
  ${module_name}/layout_util_test.cpp
  ${module_name}/shape_test.cpp
  ${module_name}/shape_util_test.cpp
  ${module_name}/sparse_tensor_test.cpp
  ${module_name}/tensor_test.cpp
  ${module_name}/util_test.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)