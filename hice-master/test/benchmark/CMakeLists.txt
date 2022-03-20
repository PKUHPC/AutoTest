set(module_name benchmark)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/spgemm_test.cpp
  ${module_name}/spmm_test.cpp
  ${module_name}/spmv_test.cpp
  ${module_name}/matmul_bench.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)