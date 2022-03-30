#include "auto_test/auto_test.h"
#include "auto_test/basic.h"


#include "hice/core/tensor.h"
#include "hice/nn/conv.h"
#include "hice/basic/factories.h"
#include "hice/nn/activation.h"
#include "hice/math/matmul.h"
#include "hice/math/binary_expr.h"
#include "hice/core/tensor_printer.h"
#include "hice/nn/pooling.h"
#include "hice/nn/softmax.h"
#include "hice/nn/batch_norm.h"
extern "C" {
#include "src/nn/pooling.h"
#include <sys/time.h>
#include "src/math/sqrt.h"
}

namespace hice{
    const DataType hice_dtypes[10] = {DataType::make<__int8_t>(),   DataType::make<uint8_t>(),
                                        DataType::make<int16_t>(),   DataType::make<uint16_t>(),
                                        DataType::make<int32_t>(),   DataType::make<uint32_t>(),
                                        DataType::make<int64_t>(),   DataType::make<uint64_t>(),
                                        DataType::make<float>(),   DataType::make<double>()};

    std::map<std::string, int> typeMap{{"signed char",0},{"unsigned char",1},{"short",2},{"unsigned short",3},{"int",4},{"unsigned int",5},{"long",6},{"unsigned long",7},{"float",8},{"double",9}};

    inline DataType hice_int_to_dtype(int n){ return hice_dtypes[n]; }
    inline Device hice_int_to_device(int n){ return Device(DeviceType::CPU); }
    inline int hice_dtype_to_int(DataType dtype){ return typeMap[dtype.name()] ;}
    inline int hice_device_to_int(Device device){ return static_cast<int>(device.type()); }

    void hice_create(DataType dtype, Device device, int64_t *dims, int64_t ndim,
                         void *data, unsigned int len, Tensor *output) {

        ConstIntArrayRef array(dims,ndim);
        DataType type = hice_dtypes[typeMap[dtype.name()]];
        hice::Tensor tensor = hice::create(array, data, len,hice::device(kCPU).dtype(type)) ;
        *output = tensor;
    }
    void hice_resolve(Tensor input, DataType *dtype, Device *device,
                          int64_t **dims, int64_t *ndim, void **data, unsigned int *len) {
        *dtype = input.data_type();
        *device = input.device();
        ConstIntArrayRef array = input.dims();

        *dims = const_cast<int64_t *>(array.data());

        *ndim = input.ndim();


        void *data_ =const_cast<void*>(input.raw_data());
        *data = data_;
        *len = input.size() * (*dtype).size();

    }

    void hice_conv2d(
            const Tensor input,
            const Tensor filter, const int *stride,
                         const int stride_len, const int *padding, const int padding_len,
                         const int *dilation, const int dilation_len,
                         const int groups,
                         Tensor *output_ptr
                         ) {
        int64_t *out_channels = const_cast<int64_t *>(filter.dims().data());
        Tensor bias_cpu = full({(*out_channels)}, 0, dtype(kFloat).device(kCPU));
        std::vector<int64_t> padding1={};
        std::vector<int64_t> stride1={} ;
        std::vector<int64_t> dilation1 = {};
        padding1.reserve(stride_len);
        stride1.reserve(padding_len);
        dilation1.reserve(dilation_len);

        for(auto i=0;i<stride_len;i++){
              padding1.push_back(padding[i]);
          }
        for(auto i=0;i<padding_len;i++){
            stride1.push_back(stride[i]);
        }
        for(auto i=0;i<dilation_len;i++){
            dilation1.push_back(dilation[i]);
        }
        *output_ptr = conv_fwd(input, filter, bias_cpu, padding1, stride1, dilation1, groups, false, false);

    }
    void hice_pooling(
            const Tensor input,const int *stride,
            const int stride_len, const int *padding, const int padding_len,
            const int *dilation, const int dilation_len,
            const int *ksize, const int ksize_len,
            const char* mode, const int mode_len,
            Tensor *output_ptr
    ){
        std::string mode_str(mode);
        std::vector<int64_t> padding1={};
        std::vector<int64_t> stride1={} ;
        std::vector<int64_t> ksize1 = {};

        padding1.reserve(stride_len);
        stride1.reserve(padding_len);
        ksize1.reserve(ksize_len);

        for(auto i=0;i<stride_len;i++){
            padding1.push_back(padding[i]);
        }
        for(auto i=0;i<padding_len;i++){
            stride1.push_back(stride[i]);
        }

        for(auto i=0;i<ksize_len;i++){
            ksize1.push_back(ksize[i]);
        }
        if(mode_str == "avg"){
            *output_ptr = pooling_avg_fwd(input, ksize1 , stride1, padding1);
        } else if(mode_str == "max"){
            auto cpu_result = pooling_max_fwd(input, ksize1 , stride1, padding1);
            Tensor cpu_output = std::get<0>(cpu_result);
            Tensor cpu_indices = std::get<1>(cpu_result);

            *output_ptr = cpu_output;

        }

    }

    void hice_softmax(const Tensor input,const int axis,
                      Tensor *output_ptr){
        *output_ptr = softmax_fwd(input,axis);
    }

    void hice_relu(const Tensor input, Tensor *output){
        *output = relu_fwd(input);
    }
    void hice_sigmoid(const Tensor input, Tensor *output){
        *output = sigmoid_fwd(input);
    }
    void hice_tanh(const Tensor input, Tensor *output){
        *output = tanh_fwd(input);
    }
    void hice_sqrt(const Tensor input, Tensor *output){
        *output = sqrt_fwd(input);
    }

    void hice_matmul(const Tensor tensor1, const Tensor tensor2,
                     Tensor *output){
        * output = matmul(tensor1, tensor2);
    }

    void hice_add(const Tensor tensor1, const Tensor tensor2, Tensor *output){
        *output = add(tensor1,tensor2);
    }

    void hice_sub(const Tensor tensor1, const Tensor tensor2, Tensor *output){
        *output = sub(tensor1,tensor2);
    }
    void hice_mul(const Tensor tensor1, const Tensor tensor2, Tensor *output){
        *output = mul(tensor1,tensor2);
    }
    void hice_div(const Tensor tensor1, const Tensor tensor2, Tensor *output){
        *output = div(tensor1,tensor2);
    }
    void hice_batchnorm(Tensor &input, int axis, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
                         Tensor &running_var,  double epsilon, Tensor &output, Tensor &bn_mean,
                        Tensor &bn_var){

        auto result = batch_norm_fwd(input, bn_scale , bn_bias, running_mean,running_var, false,2,1,epsilon,output,bn_mean,bn_var);
    }

}
REGISTER_BASIC(hice::Tensor,hice::DataType, hice::hice_int_to_dtype,hice::hice_dtype_to_int,hice::Device, hice::hice_int_to_device,
               hice::hice_device_to_int, hice::hice_create, hice::hice_resolve);

REGISTER_CONV2D(hice::hice_conv2d);

REGISTER_ACTIVATION(hice::hice_relu,hice::hice_sigmoid,hice::hice_tanh,hice::hice_sqrt);

REGISTER_MATMUL(hice::hice_matmul);

REGISTER_BINARY_OP(hice::hice_add, hice::hice_sub, hice::hice_mul,hice::hice_div);

REGISTER_POOLING(hice::hice_pooling);

REGISTER_SOFTMAX(hice_softmax);

REGISTER_BATCHNORM(hice::hice_batchnorm);

int main(int argc, char **argv){
    PERFORM_TEST;
    return 0;
}