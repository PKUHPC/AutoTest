#include "hice/hice_c.h"

#include <stdio.h>

// There is no need to run this test, it's just nice to know that it compiles.
int main(int argc, char** argv) {
  // int64_t dims[2] = {2, 2};
  // int64_t ndim = 2;
  // float data[4] = {0.1, 0.2, 0.3, 0.4};
  // HI_Tensor tensor;
  // HI_CheckStatus(HI_Create(HI_kFloat, HI_kCPU, dims, ndim, data, 4 * sizeof(*data), &tensor));
  // HI_Print(tensor);

  printf("HICE C Interface compiles good.\n");

  return 0;
}