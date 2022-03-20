set(module_name math)

list(APPEND hice_test_${module_name}_srcs_cpu
  ${module_name}/arg_reduce_dense_test.cpp
  ${module_name}/compare_test.cpp
  ${module_name}/eltwise_test.cpp
  # ${module_name}/matmul_sparse_test.cpp
  # ${module_name}/matmul_csr_test.cpp
  ${module_name}/matmul_test.cpp
  ${module_name}/reduce_test.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)
