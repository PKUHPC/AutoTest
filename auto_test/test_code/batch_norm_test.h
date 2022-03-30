#pragma once

#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/basic/factories.h"

extern "C" {
#include "src/nn/batch_norm.h"
#include <math.h>
#include <sys/time.h>
}

void batch_norm_full_float(Tensor t, const float value) {
    int64_t size = aitisa_tensor_size(t);
    auto* data = (float *)aitisa_tensor_data(t);
    for (int i = 0; i < size; ++i) {
        data[i] = value;
    }
}


namespace aitisa_api {

    namespace {

        class Batchnorm_Input : public Unary_Input {
        public:
            Batchnorm_Input() {};
            Batchnorm_Input(int64_t ndim,  std::vector<int64_t> dims, int dtype,
                          int device,  void *data, unsigned int len, int axis, double epsilon,
                            int64_t param_ndim, std::vector<int64_t> param_dims,float value, float mean, float var):
                    Unary_Input(ndim, std::move(dims), dtype, device, data, len),
                    axis_(axis),epsilon_(epsilon),param_ndim_(param_ndim),param_dims_(nullptr),value_(value),mean_(mean),var_(var){
                int64_t spatial_len  =this->param_ndim_;
                this->param_dims_ = new int64_t[spatial_len];
                for(int64_t i=0; i<spatial_len; i++){
                    this->param_dims_[i] = param_dims[i];
                }

            }
            virtual ~Batchnorm_Input() {
                delete [] param_dims_;
            }
            Batchnorm_Input &operator=(Batchnorm_Input &right) {
                int64_t spatial_len = right.param_ndim();
                auto& left = (Unary_Input&)(*this);
                left = (Unary_Input&)right;
                this->axis_ = right.axis();
                this->epsilon_ =right.epsilon();
                this->param_ndim_ =right.param_ndim();
                this->value_ = right.value();
                this->mean_ = right.mean();
                this->var_ = right.var();
                this->param_dims_ = new int64_t[spatial_len];
                memcpy(this->param_dims_, right.param_dims(), spatial_len*sizeof(int64_t));
            }

            int axis() { return axis_; }
            double epsilon() { return epsilon_; }
            int64_t param_ndim() { return param_ndim_; }
            int64_t* param_dims() {return param_dims_; }
            float value() { return value_; }
            float mean() { return mean_; }
            float var() { return var_; }

        private:
            int axis_ = 0 ;
            double epsilon_ = 0.0;
            int64_t param_ndim_ = 0;
            int64_t *param_dims_ = nullptr;
            float value_;
            float mean_;
            float var_;

        };

    } // namespace anonymous

    template <typename InterfaceType>
    class BatchnormTest : public ::testing::Test{
    public:
        BatchnormTest():
                input0(/*ndim*/4, /*dims*/{2, 3, 2, 2}, /*dtype=double*/8,
                        /*device=cpu*/0, /*data*/nullptr, /*len*/0,
                               1,1e-5,1,{3},1,0.5,0),
                input1(/*ndim*/4, /*dims*/{200, 300, 20, 20}, /*dtype=float*/8,
                        /*device=cpu*/0, /*data*/nullptr, /*len*/0,
                               1,1e-5,1,{300},1,0.5,0){
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
        virtual ~BatchnormTest(){}
        using InputType = Batchnorm_Input;
        using UserInterface = InterfaceType;
        static void aitisa_kernel(const Tensor input, const int axis, const Tensor scale,
                                  const Tensor bias, const Tensor mean,
                                  const Tensor variance, const double epsilon,
                                  Tensor *output){
            aitisa_batch_norm(input, axis, scale, bias, mean, variance, epsilon,
                    output);

        }
        // inputs
        Batchnorm_Input input0; // Natural assigned int32 type input of CPU with InputDims1{3,3,10,6}, FilterDims2{5,3,2,2}, stride{2,2}, padding{0,0}, dilation{1,1}
        Batchnorm_Input input1; // Random assigned double type input of CUDA with InputDims1{10,3,100,124,20}, FilterDims2{10,3,5,5,5}, stride{5,5,5}, padding{0,1,0}, dilation{1,1,1}
        Batchnorm_Input *input[2] = {&input0, &input1};
        std::string input0_name = "Random float of CPU with InputDims{2, 3, 2, 2}, axis{1}, epsilon{1e-5}, param_ndim{1}, param_dims{3}, value{1}, mean{0.5}, var{0}";
        std::string input1_name = "Random float of CPU with InputDims{200, 300, 20, 20}, axis{1}, epsilon{1e-5}, param_ndim{1}, param_dims{300}, value{1}, mean{0.5}, var{0}";
        std::string *input_name[2] = {&input0_name, &input1_name};
        int ninput = 2;
    };
    TYPED_TEST_CASE_P(BatchnormTest);

    TYPED_TEST_P(BatchnormTest, TwoTests){
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
            batch_norm_full_float(aitisa_tensor,this->input[i]->value());

            AITISA_Tensor mean, variance, scale, bias;

            aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->param_dims(), this->input[i]->param_ndim(), NULL, 0, &mean);
            batch_norm_full_float(mean, this->input[i]->mean());

            aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->param_dims(), this->input[i]->param_ndim(), NULL, 0, &variance);
            batch_norm_full_float(variance, this->input[i]->var());

            aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->param_dims(), this->input[i]->param_ndim(), NULL, 0, &scale);
            batch_norm_full_float(scale, 1);

            aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->param_dims(), this->input[i]->param_ndim(), NULL, 0, &bias);
            batch_norm_full_float(bias, 0);

            gettimeofday(&aitisa_start,NULL);

            aitisa_batch_norm(aitisa_tensor, this->input[i]->axis(), scale, bias, mean, variance, this->input[i]->epsilon(),
                              &aitisa_result);

            gettimeofday(&aitisa_end,NULL);
            aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0
                          + (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
            aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims,
                           &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);

            // user
            hice::TensorPrinter tp;
            UserDataType user_dtype = UserFuncs::user_int_to_dtype(this->input[i]->dtype());
            UserDevice user_device = UserFuncs::user_int_to_device(this->input[i]->device());
            UserFuncs::user_create(user_dtype, user_device, this->input[i]->dims(),
                                   this->input[i]->ndim(), this->input[i]->data(),
                                   this->input[i]->len(), &user_tensor);

            gettimeofday(&user_start,NULL);
            std::vector<int64_t> param_dims={};
            for(int k=0;k<this->input[i]->param_ndim();k++){
                param_dims.push_back(this->input[i]->param_dims()[k]);
            }
            UserTensor bn_scale = hice::empty(param_dims, device(hice::kCPU).dtype(hice::kFloat));
            UserTensor bn_bias =  hice::empty(param_dims,  device(hice::kCPU).dtype(hice::kFloat));
            UserTensor running_mean =  hice::empty(param_dims,  device(hice::kCPU).dtype(hice::kFloat));
            UserTensor running_var =  hice::empty(param_dims,   device(hice::kCPU).dtype(hice::kFloat));

            UserTensor bn_mean =  hice::full(param_dims, this->input[i]->mean(),  device(hice::kCPU).dtype(hice::kFloat));
            UserTensor bn_var =  hice::full(param_dims, this->input[i]->var(),  device(hice::kCPU).dtype(hice::kFloat));

            UserFuncs::user_create(user_dtype, user_device, this->input[i]->dims(),
                                   this->input[i]->ndim(), NULL,
                                   this->input[i]->len(), &user_result);

            UserFuncs::user_batchnorm(user_tensor, this->input[i]->axis(),bn_scale,bn_bias,running_mean,running_var,this->input[i]->epsilon(), user_result,bn_mean,bn_var);

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
                ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
            }
            ASSERT_EQ(aitisa_result_len, user_result_len);
            float *aitisa_data = (float*)aitisa_result_data;
            float *user_data = (float*)user_result_data;
            for(int64_t j=0; j<tensor_size; j++){
                printf("%f  * %f" ,aitisa_data[j] , user_data[j]);
                ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
            }
            // print result of test
            std::cout<< /*GREEN <<*/ "[ BN sample"<< i << " / "
                     << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
            std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
            std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
        }
    }
    REGISTER_TYPED_TEST_CASE_P(BatchnormTest, TwoTests);

#define REGISTER_BATCHNORM(BATCHNORM)                                                                 \
  class Batchnorm : public Basic {                                                                    \
  public:                                                                                             \
    static void user_batchnorm(UserTensor input, const int axis,  UserTensor scale,                   \
                                 UserTensor bias,  UserTensor running_mean,                           \
                                 UserTensor running_variance, const double epsilon, UserTensor output,\
                                 UserTensor mean,UserTensor var){                                     \
          BATCHNORM(input, axis, scale, bias ,running_mean,                                           \
                            running_variance,epsilon, output, mean,var );                             \
        }                                                                                             \
    };                                                                                                \
  namespace aitisa_api{                                                                               \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, BatchnormTest, Batchnorm);                              \
  }

} // namespace aitisa_api


