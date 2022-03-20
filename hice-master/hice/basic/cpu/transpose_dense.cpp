

// The following version of transpose is implemented by Ye Zilingfeng.
#if 0

#include "hice/device/cpu/context_cpu.h"
#include "hice/core/shape_util.h"
#include "hice/basic/factories.h"
#include "hice/basic/reshape.h"
#include "hice/basic/transpose.h"
#include <vector>
#include <iostream>

namespace hice {

namespace {

struct TransposePlan{
  int start, end; // start/end of index of the corresponding dimension
  int inc;  // inc step, normally set to 1. 
            // This argument is designed for "blocking scheme",
            // where the Tensor might be seperated into different blocks
            // to optimize the read-write process. Not implemented yet.
  int strideA,strideB; // strideA is the stride for input Tensor; strideB for output.
                       // help us find the next location to read/write.
  TransposePlan(){}
  TransposePlan(int s,int e,int _inc,int sA,int sB):
      start(s),end(e),inc(_inc),strideA(sA),strideB(sB){}
  void set(int s,int e,int _inc,int sA,int sB)
      {start=s; end=e; inc=_inc; strideA=sA; strideB=sB;}
};


template<class scalar_t>
void _transpose(scalar_t *in_ptr, scalar_t *out_ptr, int depth, std::vector<TransposePlan>& plans){
  if(depth == plans.size()-1 ){
    for(int i= plans[depth].start; i<plans[depth].end; i+=plans[depth].inc){
        out_ptr[i*plans[depth].strideB] = in_ptr[i*plans[depth].strideA];
    }
  }else{
    for(int i= plans[depth].start; i<plans[depth].end; i+=plans[depth].inc){
        _transpose(in_ptr+i*plans[depth].strideA,out_ptr+i*plans[depth].strideB,depth+1,plans);
    }
  }
}

void transpose_dense_impl(const Tensor& input, ConstIntArrayRef perm_dims, Tensor & output){
  // Step 0: sanity check.
  ScalarType sc_type_input = input.scalar_type();
  ScalarType sc_type_output = output.scalar_type();
  HICE_CHECK_EQ(sc_type_input, sc_type_output)
          << "Both scalar types of arguments to transpose_cpu must be equal";
  HICE_CHECK_EQ(perm_dims.size(), input.dims().size()) 
          << "The permutated dimension is "<<perm_dims.size()<<"-d array "
          << "and doesn't match with input tensor size. (Expected "
          << input.dims().size()<<"d)";
  for(int i = 0 ; i < perm_dims.size();++i)
    HICE_CHECK_LT(perm_dims[i],perm_dims.size())
          << "The permuation dimension value is incorrect, expected values less than"
          << perm_dims.size()<< " but got "<<perm_dims[i]<<" at position "<< i ;
  
  ScalarType sc_type = sc_type_input;
  HICE_DISPATCH_ALL_TYPES(sc_type, "transpose_cpu", [&]() {
    // Implementaion based on a C-style array layout(row first)
    if(input.ndim()==1){
      output = input;
    }else if(input.ndim()==2){
      if(perm_dims[0] == 1){
        int64_t stride_out = input.dims()[0];
        int64_t stride_in = input.dims()[1];
        output = reshape_(output,{input.dims()[1],input.dims()[0]});

        auto out_ptr = output.mutable_data<scalar_t>();
        auto in_ptr = input.data<scalar_t>();
        // To-do : accelerate the following for-loop part using parallelism
        for(int i = 0 ;i < output.dims()[0];++i){
            for(int j = 0;j < output.dims()[1];++j){
                out_ptr[i * stride_out + j] = in_ptr[j * stride_in + i]; 
            }
        }
      }else{ // perm_dims[0] == 0 meaning no actual transpose will be performed
        output = input;
      }
    }else{// Transpose of high dimensional Tensor
      std::vector<int64_t> output_dims(input.ndim());
      for(int i = 0 ;i < input.ndim();++i){
          output_dims[i] = input.dims()[perm_dims[i]];
      }
      output = reshape_(output,ConstIntArrayRef(output_dims));

      // Step 1: create TransposePlan for transpose:
      // Each "plan" correspond to one layer of nested loop,
      // it describes the start and end, as well as strides, of the loop.
      // To-do: carefully arrange the order of the Plans to accelerate read and write.
      //        (!! The process to arrange might deteriorate the efficiency.)
      std::vector<TransposePlan> plans(input.ndim());
      for(int i = 0 ;i < input.ndim();++i){
          plans[i].set(0, output.dims()[i], 1, input.strides()[perm_dims[i]], output.strides()[i]);
      }
      // Step 2: transpose according to TransposePlan, function "_transpose" works
      // in a recursive manner, similar to the nested loop manner.
      auto out_ptr = output.mutable_data<scalar_t>();
      auto in_ptr = input.data<scalar_t>();
      _transpose(in_ptr, out_ptr, 0, plans);
    }
  });
}

} // anonymous namespace

HICE_REGISTER_KERNEL(transpose_dispatcher, &transpose_dense_impl,
                     {kCPU, kDense}, // const Tensor & input
                     {kCPU, kDense}  // Tensor & output
);
} // namespace hice

#endif