#pragma once

#include <random>

#ifdef HICE_USE_MKLDNN
#include "common_mkldnn.h"
#endif // HICE_USE_MKLDNN

#include "hice/core/context.h"

namespace hice {

// Thread-local cpu objects used by cpu context
class ThreadLocalCPUObjects {
 public:
  friend class CPUContext;

 private:
  ThreadLocalCPUObjects() {}

#ifdef HICE_USE_MKLDNN
        dnnl::engine mkldnn_engine() {
    return mkldnn_engine_;
  }

        dnnl::stream mkldnn_stream() {
    return mkldnn_stream_;
  }
#endif

 private:
#ifdef HICE_USE_MKLDNN
        dnnl::engine mkldnn_engine_ = {dnnl::engine::kind::cpu, 0};
        dnnl::stream mkldnn_stream_ = {mkldnn_engine_};
#endif
};

class CPUContext: public DeviceContext {
 public:
  typedef std::mt19937 RandGenType;
  CPUContext() : random_seed_(std::random_device{}()) {}

  explicit CPUContext(const Device& device)
      : random_seed_(std::random_device{}()) {}

  ~CPUContext() override {}

  Device device() const override {
    return Device(DeviceType::CPU);
  }

  DeviceType device_type() const override {
    return DeviceType::CPU;
  }

  void switch_to_device(int /*stream_id*/) override {}

  void synchronize() override {}

  RandGenType& rand_generator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new RandGenType(random_seed_));
    }
    return *random_generator_.get();
  }

#ifdef HICE_USE_MKLDNN
        dnnl::engine mkldnn_engine() {
    return cpu_objects().mkldnn_engine();
  }
        dnnl::stream mkldnn_stream() {
    return cpu_objects().mkldnn_stream();
  }
#endif

 private:
#ifdef HICE_USE_MKLDNN
  static ThreadLocalCPUObjects& cpu_objects() {
    static thread_local ThreadLocalCPUObjects cpu_objects_;
    return cpu_objects_;
  }
#endif

  int random_seed_;
  std::unique_ptr<RandGenType> random_generator_;
};

} //namespace hice
