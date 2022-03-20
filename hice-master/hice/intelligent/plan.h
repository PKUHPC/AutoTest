#pragma once

#include <hice/core/export.h>
#include <hice/core/tensor.h>
#include <hice/util/dlpack_wrapper.h>

#include <memory>
#include <vector>
#include <limits>


#ifndef HICE_PLAN_EVAL
#define HICE_PLAN_EVAL(func_hice, func_tvm)  \
  if (!is_tvm_available()) HICE_LOG(ERROR) << "tvm is not enabled"; \
  if (is_evaluated_) return;                                        \
  HICE_DLOG(INFO) << "search tvm kernel...";                        \
  bool success = func_tvm();                                        \
  HICE_DLOG(INFO) << "bench tvm kernel...";                         \
  float tvm_time = 0.0;                                             \
  if (success) {                                                    \
    tvm_time = bench_cpu(func_tvm);                                 \
  } else {                                                          \
    tvm_time = std::numeric_limits<float>::max();                   \
  }                                                                 \
  HICE_DLOG(INFO) << "bench hice kernel...";                        \
  float hice_time = bench_cpu(func_hice);                           \
  HICE_DLOG(INFO) << "tvm_time: " << tvm_time;                      \
  HICE_DLOG(INFO) << "hice_time: " << hice_time;                    \
  impl_type_ = hice_time > tvm_time ? kTVM : kOfficial;             \
  is_evaluated_ = true;        
#endif

#ifndef HICE_PLAN_EXEC
#define HICE_PLAN_EXEC(func_hice, func_tvm)     \
  if (impl_type_ == kTVM) {                     \
    HICE_DLOG(INFO) << "execute tvm kernel.";   \
    func_tvm();                                 \
  } else {                                      \
    HICE_DLOG(INFO) << "execute hice kernel.";  \
    func_hice();                                \
  }                                             
#endif

#ifndef REGISTER_PLAN_IN_HICETENSOR
#define REGISTER_PLAN_IN_HICETENSOR(in)    \
  inputs_.push_back(in);                                   \
  dl_inputs_.push_back(HICETensor_to_DLTensor(in));        
#endif

#ifndef REGISTER_PLAN_IN
#define REGISTER_PLAN_IN(in)    \
  dl_inputs_.push_back(in);        \
  inputs_.push_back(DLTensor_to_HICETensor(in));
#endif

#ifndef REGISTER_PLAN_OUT
#define REGISTER_PLAN_OUT(out)    \
  dl_outputs_.push_back(out);        \
  outputs_.push_back(DLTensor_to_HICETensor(out));
#endif

namespace hice {

typedef std::vector<DLTensor> DLTensorArray;
typedef std::vector<hice::Tensor> TensorArray;

enum HICE_API ImplementationType {
  kNative = 0,
  kOfficial,
  kTVM,
  kUnKnown
};

class HICE_API Plan {
public:
  
  TensorArray& inputs() { return inputs_; }
  TensorArray& outputs() { return outputs_; }
  DLTensorArray& dl_inputs() { return dl_inputs_; }
  DLTensorArray& dl_outputs() { return dl_outputs_; }
  
  Tensor& input(int i) { return inputs_[i]; }
  Tensor& output(int i) { return outputs_[i]; }
  DLTensor& dl_input(int i) { return dl_inputs_[i]; }
  DLTensor& dl_output(int i) { return dl_outputs_[i]; }

  bool is_evaluated() const { return is_evaluated_; }
  ImplementationType impl_type() const { return impl_type_; }

  void set_evaluated(bool eval) { is_evaluated_ = eval; }
  void set_impl_type(ImplementationType type) { impl_type_ = type; }

  void update_input_dataptr(int idx, void* data_ptr) {
    if (dl_inputs_[idx].data == data_ptr) return;
    dl_inputs_[idx].data = data_ptr;
    inputs_[idx].mutable_impl()
                .mutable_storage()
                .set_data_ptr(hice::DataPtr(static_cast<void *>(data_ptr),
                                            static_cast<void *>(data_ptr),
                                            [](void*){}));
  }

  void update_output_dataptr(int idx, void* data_ptr) {
    if (dl_outputs_[idx].data == data_ptr) return;
    dl_outputs_[idx].data = data_ptr;
    outputs_[idx].mutable_impl()
                .mutable_storage()
                .set_data_ptr(hice::DataPtr(static_cast<void *>(data_ptr),
                                            static_cast<void *>(data_ptr),
                                            [](void*){}));
  }

  virtual void evaluate() = 0;
  virtual void execute() = 0;

  virtual ~Plan() {}

/// FIXME: should be protected, here just for test.
public:
  TensorArray inputs_;
  TensorArray outputs_;
  DLTensorArray dl_inputs_;
  DLTensorArray dl_outputs_;
  bool is_evaluated_ = false;
  ImplementationType impl_type_ = kOfficial;
};  // class Plan

typedef std::shared_ptr<Plan> PlanPtr;

template<typename T, typename... Args>
PlanPtr make_plan(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

} // namespace hice