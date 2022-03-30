#pragma once

#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/core/tensor_printer.h"

extern "C" {
#include <math.h>
#include <sys/time.h>
#include "src/nn/dropout.h"
}

namespace aitisa_api {

    namespace {

        class Dropout_Input : public Unary_Input {
        public:
            Dropout_Input() = default;;
            Dropout_Input(int64_t ndim, std::vector<int64_t> dims, int dtype,
                          int device, void *data, unsigned int len, double rate, int initvalue) :
                    Unary_Input(ndim, dims, dtype, device, data, len), rate_(rate), initvalue_(initvalue) {}

            ~Dropout_Input() override = default;

            Dropout_Input &operator=(Dropout_Input &right) {
                auto &left = (Unary_Input &) (*this);
                left = (Unary_Input &) right;
                this->rate_ = right.rate();
                this->initvalue_ = right.initvalue();
            }
            double rate() { return rate_; }
            int initvalue() { return initvalue_; }
        private:
            double rate_ = 0;
            int initvalue_ = 0;
        };
    } // namespace anonymous


    template <typename InterfaceType>
    class DropoutTest : public ::testing::Test{
    public:
        DropoutTest():
                input0(/*ndim*/3, /*dims*/{256, 256, 256}, /*dtype=float*/8,
                        /*device=cpu*/0, /*data*/nullptr, /*len*/0, 0.5,1),
                input1(/*ndim*/4, /*dims*/{64, 64, 64, 64}, /*dtype=float*/8,
                        /*device=cpu*/0, /*data*/nullptr, /*len*/0, 0.7,1){
            input[0] = &input0;
            input[1] = &input1;
            ninput = 2;
            for(int i=0; i<ninput; i++){
                unsigned int input_nelem = 1;
                for(unsigned int j=0; j<input[i]->ndim(); j++){
                    input_nelem *= input[i]->dims()[j];
                }

                unsigned int input_len = input_nelem * elem_size(input[i]->dtype());
                void *input_data = (void*) new char[input_len];
                random_assign(input_data, input_len, input[i]->dtype());
                input[i]->set_data(input_data, input_len);
            }
        }
        virtual ~DropoutTest(){}
        using InputType = Dropout_Input;
        using UserInterface = InterfaceType;
        static void aitisa_kernel(const AITISA_Tensor input, const double rate,
                                  AITISA_Tensor *output){
            aitisa_dropout(input,rate,output);
        }
        // inputs
        Dropout_Input input0;
        Dropout_Input input1;
        Dropout_Input *input[2] = {&input0, &input1};
        std::string input0_name = "Full float of CPU with InputDims{256, 256,256}, rate{0.5}";
        std::string input1_name = "Full float of CPU with InputDims{64, 64, 64, 64}, rate{0.7}";
        std::string *input_name[2] = {&input0_name, &input1_name};
        int ninput = 2;
    };
    TYPED_TEST_CASE_P(DropoutTest);

    TYPED_TEST_P(DropoutTest, TwoTests){
        using UserDataType = typename TestFixture::UserInterface::UserDataType;
        using UserDevice = typename TestFixture::UserInterface::UserDevice;
        using UserTensor = typename TestFixture::UserInterface::UserTensor;
        using UserFuncs = typename TestFixture::UserInterface;
        for(int i=0; i<this->ninput; i++){
            struct timeval aitisa_start, aitisa_end, user_start, user_end;
            double aitisa_time, user_time;
            int64_t aitisa_result_ndim, user_result_ndim;
            int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
            float *aitisa_result_data=nullptr, *user_result_data=nullptr;
            unsigned int aitisa_result_len, user_result_len;
            AITISA_Tensor aitisa_tensor, aitisa_result;
            AITISA_DataType aitisa_result_dtype;
            AITISA_Device aitisa_result_device;
            UserTensor user_tensor, user_result;
            UserDataType user_result_dtype;
            UserDevice user_result_device;
            // aitisa
            AITISA_DataType aitisa_dtype = aitisa_int_to_dtype(this->input[i]->dtype());
            AITISA_Device aitisa_device = aitisa_int_to_device(0); // cpu supoorted only
            aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->dims(), this->input[i]->ndim(),
                          (void*)(this->input[i]->data()), this->input[i]->len(), &aitisa_tensor);
            full_float(aitisa_tensor, this->input[i]->initvalue());

            gettimeofday(&aitisa_start,NULL);

            aitisa_dropout(aitisa_tensor,this->input[i]->rate(),&aitisa_result);

            gettimeofday(&aitisa_end,NULL);
            aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0
                          + (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
            aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims,
                           &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);



            // user
            UserDataType user_dtype = UserFuncs::user_int_to_dtype(this->input[i]->dtype());
            UserDevice user_device = UserFuncs::user_int_to_device(this->input[i]->device());
            UserFuncs::user_create(user_dtype, user_device, this->input[i]->dims(),
                                   this->input[i]->ndim(), this->input[i]->data(),
                                   this->input[i]->len(), &user_tensor);
            gettimeofday(&user_start,NULL);

            UserFuncs::user_dropout(user_tensor, this->input[i]->rate(), &user_result);

            gettimeofday(&user_end,NULL);
            user_time = (user_end.tv_sec - user_start.tv_sec) * 1000.0
                        + (user_end.tv_usec - user_start.tv_usec) / 1000.0;
            UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device,
                                    &user_result_dims, &user_result_ndim,
                                    (void**)&user_result_data, &user_result_len);

            // compare
            int64_t tensor_size = 1;
            ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
            ASSERT_EQ(/*CUDA*/0, UserFuncs::user_device_to_int(user_result_device));
            ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                      UserFuncs::user_dtype_to_int(user_result_dtype));
            for(int64_t j=0; j<aitisa_result_ndim; j++){
                tensor_size *= aitisa_result_dims[j];
                ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
            }
            ASSERT_EQ(aitisa_result_len, user_result_len);

            float *aitisa_data = (float*)aitisa_result_data;
            float *user_data = (float*)user_result_data;
            int aitisa_count = 0;
            int user_count = 0;
            for(int64_t j=0; j<tensor_size; j++){
                if(aitisa_data[j]==0){
                    aitisa_count++;
                }
                if(user_data[j]==0){
                    user_count++;
                }
            }
            ASSERT_TRUE(abs((double )aitisa_count/(double )tensor_size - (double )user_count/(double )tensor_size) < 0.2);

            // print result of test
            std::cout<< /*GREEN <<*/ "[ DropOut sample"<< i << " / "
                     << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
            std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
            std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
        }
    }
    REGISTER_TYPED_TEST_CASE_P(DropoutTest, TwoTests);

#define REGISTER_DROPOUT(DROPOUT)                                                               \
  class Dropout : public Basic {                                                                \
  public:                                                                                       \
  static void user_dropout(UserTensor input, const double rate,                                 \
                            UserTensor *output){                                                \
      DROPOUT(input, rate ,output);                                                             \
    }};                                                                                         \
  namespace aitisa_api{                                                                         \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, DropoutTest, Dropout);                            \
  }

} // namespace aitisa_api


