set(module_name api_c)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/ops_math_test_c.cpp
  ${module_name}/ops_nn_test_c.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)


# Test whether C interface headers can be complied by C-compiler
add_executable(compile_test_c compile_test_c.c)
target_link_libraries(compile_test_c "hice::hice")
