#ifndef IMAGE_NORMALIZE
#define IMAGE_NORMALIZE

#include "src/core/tensor.h"

/**
 * @brief Normalize the input image according to the given mean and standard deviation.
 *
 * @param input The input tensor.
 * @param mean The mean of each channel of the input image.
 * @param std The standard deviation of each channel of the input image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_image_normalize(const Tensor input, double *mean, double *std, Tensor *output);

#endif // IMAGE_NORMALIZE
