set(module_name basic)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/copy_test.cpp
  ${module_name}/factories_test.cpp
  ${module_name}/reshape_test.cpp
  ${module_name}/resize_test.cpp
  ${module_name}/slice_test.cpp
  ${module_name}/split_test.cpp
  ${module_name}/transpose_test.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)