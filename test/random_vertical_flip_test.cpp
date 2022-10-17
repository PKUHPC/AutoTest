#include "gtest/gtest.h"
extern "C"
{
#include "src/new_ops1/random_vertical_flip.h"
    // #include "src/tool/tool.h"
}

void random_vflip_assign_float(Tensor t)
{
    int64_t size = aitisa_tensor_size(t);
    float *data = (float *)aitisa_tensor_data(t);
    float value = 0;
    for (int i = 0; i < size; ++i)
    {
        value = i * 0.1 - 0.3;
        data[i] = value;
    }
}

namespace aitisa_api
{
    namespace
    {

        TEST(RandomVflip, Float3d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[3] = {2, 2, 3};
            aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
            random_vflip_assign_float(input);
            const float prob = 0.8;
            const int seed = 0;
            // tensor_printer2d(input);

            Tensor output;
            aitisa_random_vertical_flip(input, prob, seed, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data1[] = {-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
            float test_data2[] = {0, 0.1, 0.2, -0.3, -0.2, -0.1, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5};
            int rate_line = (int)(100 * prob);

            srand(seed);
            if (rand() % 100 + 1 < rate_line)
            {
                int64_t size = aitisa_tensor_size(input);
                for (int64_t i = 0; i < size; i++)
                {
                    /* Due to the problem of precision, consider the two numbers
                    are equal when their difference is less than 0.000001*/
                    EXPECT_TRUE(abs(out_data[i] - test_data2[i]) < 0.000001);
                }
            }
            else
            {
                int64_t size = aitisa_tensor_size(input);
                for (int64_t i = 0; i < size; i++)
                {
                    /* Due to the problem of precision, consider the two numbers
                    are equal when their difference is less than 0.000001*/
                    EXPECT_TRUE(abs(out_data[i] - test_data1[i]) < 0.000001);
                }
            }

            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }

        TEST(RandomVflip, Float4d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[4] = {2, 2, 2, 3};
            aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
            random_vflip_assign_float(input);
            const float prob = 0.9;
            const int seed = 0;
            // tensor_printer2d(input);

            Tensor output;
            aitisa_random_vertical_flip(input, prob, seed, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data1[] = {-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                  0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
            float test_data2[] = {0, 0.1, 0.2, -0.3, -0.2, -0.1, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5,
                                  1.2, 1.3, 1.4, 0.9, 1.0, 1.1, 1.8, 1.9, 2.0, 1.5, 1.6, 1.7};
            // float test_data2[] = {0, 0.1, 0.2, -0.3, -0.2, -0.1, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5,
            //                       0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
            // float test_data2[] = {-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            //                       1.2, 1.3, 1.4, 0.9, 1.0, 1.1, 1.8, 1.9, 2.0, 1.5, 1.6, 1.7};
            int rate_line = (int)(100 * prob);

            srand(seed);
            
            if (rand() % 100 + 1 < rate_line)
            {
                int64_t size = aitisa_tensor_size(input);
                for (int64_t i = 0; i < size; i++)
                {
                    /* Due to the problem of precision, consider the two numbers
                    are equal when their difference is less than 0.000001*/
                    EXPECT_TRUE(abs(out_data[i] - test_data2[i]) < 0.000001);
                }
            }
            else
            {
                int64_t size = aitisa_tensor_size(input);
                for (int64_t i = 0; i < size; i++)
                {
                    /* Due to the problem of precision, consider the two numbers
                    are equal when their difference is less than 0.000001*/
                    EXPECT_TRUE(abs(out_data[i] - test_data1[i]) < 0.000001);
                }
            }
            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }
    } // namespace
} // namespace aitisa_api