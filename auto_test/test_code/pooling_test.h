#pragma once

#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include "src/nn/pooling.h"
#include <math.h>
#include <sys/time.h>
}

namespace aitisa_api {

namespace {

    class Pooling_Input : public Unary_Input {
    public:
        Pooling_Input() {};
        Pooling_Input(int64_t ndim, int64_t *dims, int dtype,
                   int device, void *data, unsigned int len,
                   int *stride, int *padding, int *dilation, int *ksize, char* mode):
                Unary_Input(ndim, dims, dtype, device, data, len),
                stride_(stride), padding_(padding), dilation_(dilation), ksize_(ksize),mode_(mode){}
        Pooling_Input(int64_t ndim,  std::vector<int64_t> dims, int dtype,
                   int device,  void *data, unsigned int len,std::vector<int> stride, std::vector<int> padding,
                      std::vector<int> dilation,std::vector<int> ksize, std::string mode):
                Unary_Input(ndim, dims, dtype, device, data, len),
                stride_(nullptr), padding_(nullptr), dilation_(nullptr), ksize_(nullptr), mode_(nullptr) {
            int spatial_len = ndim - 2;
            this->stride_ = new int[spatial_len];
            this->padding_ = new int[spatial_len];
            this->dilation_ = new int[spatial_len];
            this->ksize_ = new int[spatial_len];
            this->mode_ = new char[4];
            for(int i=0;i<3;i++){
                this->mode_[i]= mode[i];
            }
            mode_[3]='\0';
            for(int i=0; i<spatial_len; i++){
                this->stride_[i] = stride[i];
                this->padding_[i] = padding[i];
                this->dilation_[i] = dilation[i];
                this->ksize_[i] = ksize[i];
            }
        }
        virtual ~Pooling_Input() {
            delete [] stride_;
            delete [] padding_;
            delete [] dilation_;
            delete [] ksize_;
            delete [] mode_;
        }
        Pooling_Input & operator=(Pooling_Input& right) {
            int spatial_len = right.ndim() - 2;
            Unary_Input& left = (Unary_Input&)(*this);
            left = (Unary_Input&)right;
            this->stride_ = new int[spatial_len];
            this->padding_ = new int[spatial_len];
            this->dilation_ = new int[spatial_len];
            this->ksize_ = new int[spatial_len];
            this->mode_ = new char[3];
            memcpy(this->stride_, right.stride(), spatial_len*sizeof(int));
            memcpy(this->padding_, right.padding(), spatial_len*sizeof(int));
            memcpy(this->dilation_, right.dilation(), spatial_len*sizeof(int));
            memcpy(this->ksize_,right.ksize(),spatial_len*sizeof(int));
            memcpy(this->mode_,right.mode(),3*sizeof(char));
        }
        int* stride() { return stride_; }
        int* padding() { return padding_; }
        int* dilation() { return dilation_; }
        int* ksize() { return ksize_; }
        char* mode() {return mode_;}
    private:
        int *stride_ = nullptr;
        int *padding_ = nullptr;
        int *dilation_ = nullptr;
        int *ksize_ = nullptr;
        char *mode_ = nullptr;
    };

} // namespace anonymous

template <typename InterfaceType>
class PoolingTest : public ::testing::Test{
public:
    PoolingTest():
            input0(/*ndim*/4, /*dims*/{50, 30, 50, 40}, /*dtype=float*/8,
                    /*device=cpu*/0, /*data*/nullptr, /*len*/0,
                    /*stride*/{1, 1}, /*padding*/{0,0}, /*dilation*/{1,1},
                    /*ksize*/{2, 2},"avg"),
            input1(/*ndim*/4, /*dims*/{3, 2, 4, 6}, /*dtype=float*/8,
                    /*device=cuda*/0, /*data*/nullptr, /*len*/0,
                    /*stride*/{2, 2}, /*padding*/{0,0}, /*dilation*/{1,1},
                    /*ksize*/{3, 2},"max"){
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
    virtual ~PoolingTest(){}
    using InputType = Pooling_Input;
    using UserInterface = InterfaceType;
    static void aitisa_kernel(const AITISA_Tensor input, const char *mode,
                              const int *ksize,	  const int *stride,
                              const int *padding, const int *dilation,
                              AITISA_Tensor *output){
        aitisa_pooling(input,mode,ksize,stride,padding,dilation,output);

    }
    // inputs
    Pooling_Input input0; // Natural assigned int32 type input of CPU with InputDims1{3,3,10,6}, FilterDims2{5,3,2,2}, stride{2,2}, padding{0,0}, dilation{1,1}
    Pooling_Input input1; // Random assigned double type input of CUDA with InputDims1{10,3,100,124,20}, FilterDims2{10,3,5,5,5}, stride{5,5,5}, padding{0,1,0}, dilation{1,1,1}
    Pooling_Input *input[2] = {&input0, &input1};
    std::string input0_name = "Random float of CPU with InputDims{50, 30, 50, 40}, ksize{2, 2}, stride{2,2}, padding{0,0}, dilation{1,1}, mode{avg}";
    std::string input1_name = "Random float of CPU with InputDims{3, 2, 4, 6}, ksize{3, 2}, stride{3,3}, padding{0,0}, dilation{1,1}, mode{max}";
    std::string *input_name[2] = {&input0_name, &input1_name};
    int ninput = 2;
};
TYPED_TEST_CASE_P(PoolingTest);

TYPED_TEST_P(PoolingTest, TwoTests){
    using UserDataType = typename TestFixture::UserInterface::UserDataType;
    using UserDevice = typename TestFixture::UserInterface::UserDevice;
    using UserTensor = typename TestFixture::UserInterface::UserTensor;
    using UserFuncs = typename TestFixture::UserInterface;
    for(int i=0; i<this->ninput; i++){
//             if(i==0) continue;
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
        gettimeofday(&aitisa_start,NULL);



        aitisa_pooling(aitisa_tensor,this->input[i]->mode(),this->input[i]->ksize(),this->input[i]->stride(),this->input[i]->padding(),this->input[i]->dilation(),&aitisa_result);

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
        UserFuncs::user_pooling(user_tensor, this->input[i]->stride(),2 ,this->input[i]->padding(),2,
                               this->input[i]->dilation(),2, this->input[i]->ksize(),2,this->input[i]->mode(),3, &user_result);
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
        for(int64_t j=0; j<tensor_size; j++){
//            printf("%f * %f\t", aitisa_data[j] , user_data[j]);
                ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
        }
        // print result of test
        std::cout<< /*GREEN <<*/ "[ Pooling sample"<< i << " / "
                 << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
        std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
        std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
    }
}
REGISTER_TYPED_TEST_CASE_P(PoolingTest, TwoTests);

#define REGISTER_POOLING(POOLING)                                                               \
  class Pooling : public Basic {                                                                \
  public:                                                                                       \
 static void user_pooling(UserTensor input, const int *stride,                                  \
                            const int stride_len, const int *padding, const int padding_len,    \
                            const int *dilation, const int dilation_len, const int *ksize,      \
                            const int ksize_len, const char *mode, const int mode_len,          \
                            UserTensor *output){                                                \
      POOLING(input, stride, stride_len, padding,                                               \
           padding_len, dilation, dilation_len, ksize, ksize_len, mode, mode_len,output);       \
    }                                                                                           \
  };                                                                                            \
  namespace aitisa_api{                                                                         \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, PoolingTest, Pooling);                            \
  }

} // namespace aitisa_api