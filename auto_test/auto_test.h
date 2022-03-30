#pragma once

#include "auto_test/test_code/binary_op_test.h"
#include "auto_test/test_code/matmul_test.h"
#include "auto_test/test_code/conv_test.h"
#include "auto_test/test_code/activation_test.h"
#include "auto_test/test_code/pooling_test.h"
#include "auto_test/test_code/softmax_test.h"
#include "auto_test/test_code/batch_norm_test.h"
#include "auto_test/test_code/drop_out_test.h"

#define REGISTER_OP(ADD, SUB, MUL, DIV, MATMUL, CONV)   \
  REGISTER_BINARY_OP(ADD, SUB, MUL, DIV);               \
  REGISTER_MATMUL(MATMUL);                              \
  REGISTER_CONV(CONV);                                  \
  REGISTER_POOLING(POOLING);                            \
  REGISTER_SOFTMAX(SOFTMAX);                            \
  REGISTER_BATCHNORM()                                  \
  REGISTER_DROPOUT()

#define PERFORM_TEST                                    \
  ::testing::InitGoogleTest(&argc, argv);               \
  return RUN_ALL_TESTS();

