#pragma once

#include "auto_test/test_code/activation_test.h"
#include "auto_test/test_code/batch_norm_test.h"
#include "auto_test/test_code/binary_op_test.h"
#include "auto_test/test_code/conv_test.h"
#include "auto_test/test_code/drop_out_test.h"
#include "auto_test/test_code/matmul_test.h"
#include "auto_test/test_code/pooling_test.h"
#include "auto_test/test_code/softmax_test.h"

#define PERFORM_TEST                      \
  ::testing::InitGoogleTest(&argc, argv); \
  return RUN_ALL_TESTS();
