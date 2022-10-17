#include "gtest/gtest.h"
extern "C"
{
#include "src/new_ops1/rot90.h"
// #include "src/tool/tool.h"
}

void rot90_assign_float(Tensor t)
{
    int64_t size = aitisa_tensor_size(t);
    float *data = (float *)aitisa_tensor_data(t);
    float value = 0;
    for (int i = 0; i < size; ++i)
    {
        value = i * 0.1;
        data[i] = value;
    }
}

namespace aitisa_api
{
    namespace
    {
        TEST(Rot1, Float2d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[2] = {2, 3};
            aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
            rot90_assign_float(input);
            // tensor_printer2d(input);

            Tensor output;
            aitisa_rot90(input, 1, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data[] = {0.2, 0.5, 0.1, 0.4, 0.0, 0.3};
            int64_t size = aitisa_tensor_size(input);
            for (int64_t i = 0; i < size; i++)
            {
                /* Due to the problem of precision, consider the two numbers
                   are equal when their difference is less than 0.000001*/
                EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
            }

            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }
        TEST(Rot2, Float2d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[2] = {2, 3};
            aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
            rot90_assign_float(input);
            // tensor_printer2d(input);

            Tensor output;
            aitisa_rot90(input, 2, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data[] = {0.5, 0.4, 0.3, 0.2, 0.1, 0};
            int64_t size = aitisa_tensor_size(input);
            for (int64_t i = 0; i < size; i++)
            {
                /* Due to the problem of precision, consider the two numbers
                   are equal when their difference is less than 0.000001*/
                EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
            }

            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }
        TEST(Rot3, Float2d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[2] = {2, 3};
            aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
            rot90_assign_float(input);
            // tensor_printer2d(input);

            Tensor output;
            aitisa_rot90(input, 3, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data[] = {0.3, 0, 0.4, 0.1, 0.5, 0.2};
            int64_t size = aitisa_tensor_size(input);
            for (int64_t i = 0; i < size; i++)
            {
                /* Due to the problem of precision, consider the two numbers
                   are equal when their difference is less than 0.000001*/
                EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
            }

            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }
        TEST(Rot8, Float2d)
        {
            Tensor input;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims[2] = {2, 3};
            aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
            rot90_assign_float(input);
            // tensor_printer2d(input);

            Tensor output;
            aitisa_rot90(input, 8, &output);
            // tensor_printer2d(output);

            float *out_data = (float *)aitisa_tensor_data(output);
            float test_data[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5};
            int64_t size = aitisa_tensor_size(input);
            for (int64_t i = 0; i < size; i++)
            {
                /* Due to the problem of precision, consider the two numbers
                   are equal when their difference is less than 0.000001*/
                EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
            }

            aitisa_destroy(&input);
            aitisa_destroy(&output);
        }
        
    } // namespace
} // namespace aitisa_api