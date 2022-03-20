#pragma once

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <type_traits>
#include <iostream>

#include "hice/core/macros.h"
#include "hice/util/traits.h"
#include "hice/core/tensor.h"

namespace hice {

constexpr int kNumTypes = kNumDeviceTypes * kNumLayoutTypes;

template <typename TOp, typename TFnPtr>
struct Dispatcher;

template <typename TOp, typename TReturn, typename... TArgs>
struct Dispatcher<TOp, TReturn (*)(TArgs...)> {
  using FnPtr = TReturn (*)(TArgs...);

  template<typename... TArgTypes>
  TReturn operator()(TArgTypes&&... args) {
    int key = dispatch_key(std::forward<TArgTypes>(args)...);
    FnPtr kernel = kernel_table_[key];
    HICE_CHECK_NOTNULL(kernel) << "Missing kernel: unsupported device type or format type";
    return (*kernel)(std::forward<TArgTypes>(args)...);
  }

  void register_kernel(int dispatch_key, FnPtr kernel) {
    std::lock_guard<std::mutex> guard(mu_);
    kernel_table_[dispatch_key] = kernel;
  }

 private:

  template<typename T>
  int dispatch_key(const T& arg) {
    return tensor_type_id(arg);
  }

  template<typename THead, typename... TTail>
  int dispatch_key(const THead& head, const TTail&... tail) {
    return tensor_type_id(head) + key_coeff(tail...) * dispatch_key(tail...);
  }

  template<typename T,
           typename ext::enable_if_t<is_tensor<T>::value, int> = 0>
  int tensor_type_id(const T& tensor) {
    return tensor.type_id();
  }

  template<typename T,
           typename ext::enable_if_t<ext::negation<is_tensor<T>>::value, int> = 0>
  int tensor_type_id(const T& non_tensor) {
    return 0;
  }

  template<typename THead,
           typename... TTail,
           typename ext::enable_if_t<is_tensor<THead>::value, int> = 0>
  int key_coeff(const THead& head, const TTail&... tail) {
    return kNumTypes;
  }

  template<typename THead,
           typename... TTail,
           typename ext::enable_if_t<ext::negation<is_tensor<THead>>::value, int> = 0>
  int key_coeff(const THead& head, const TTail&... tail) {
    return 1;
  }

  std::mutex mu_;
  std::unordered_map<int, FnPtr> kernel_table_;
};

namespace detail {

struct DispatchKey {
  DeviceType device_type;
  LayoutType layout_type;
  constexpr int tensor_type_id() const {
    return static_cast<int>(device_type)
        + kNumDeviceTypes * static_cast<int>(layout_type);
  }
};

static inline int dispatch_key(std::initializer_list<DispatchKey> init) {
  int key = 0;
  int stride = 1;
  for (auto iter = init.begin(); iter != init.end(); ++iter) {
    key += iter->tensor_type_id() * stride;
    stride *= kNumTypes;
  }
  return key;
}

} // namespace detail

template <typename TOp, typename TFnPtr>
struct KernelRegistrar {
  KernelRegistrar(Dispatcher<TOp, TFnPtr> *dispatcher,
                 TFnPtr kernel,
                 std::initializer_list<detail::DispatchKey> init) {
    int key = detail::dispatch_key(init);
    dispatcher->register_kernel(key, kernel);
  }
};

#define HICE_DECLARE_DISPATCHER(dispatcher, kernel_fn_type)              \
  struct dispatcher##_t : Dispatcher<dispatcher##_t, kernel_fn_type> {}; \
  dispatcher##_t* get_##dispatcher();                                    \
  extern dispatcher##_t& dispatcher

#define HICE_DEFINE_DISPATCHER(dispatcher)                          \
  dispatcher##_t* get_##dispatcher() {                              \
    static dispatcher##_t* dispatcher = new hice::dispatcher##_t(); \
    return dispatcher;                                              \
  }                                                                 \
  dispatcher##_t& dispatcher = *get_##dispatcher()

#define HICE_REGISTER_KERNEL(dispatcher, fn, ...)            \
  static KernelRegistrar<struct dispatcher##_t, decltype(fn)> \
      HICE_ANONYMOUS_VARIABLE(kernel_register_##dispatcher)(  \
          get_##dispatcher(), fn,                            \
          std::initializer_list<detail::DispatchKey>{__VA_ARGS__})

#define HICE_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                  \
    using scalar_t = type;                           \
    return __VA_ARGS__();                            \
  }

#define HICE_DISPATCH_INTEGRAL_TYPES(SCALAR_TYPE, NAME, ...)                 \
  [&] {                                                                       \
    switch (SCALAR_TYPE) {                                                    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt8, uint8_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int8, int8_t, __VA_ARGS__)     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt16, uint16_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int16, int16_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt32, uint32_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int32, int32_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt64, uint64_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int64, int64_t, __VA_ARGS__)   \
      default:                                                                \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                  \
                        << to_string(SCALAR_TYPE) << "'";                     \
    }                                                                         \
  }()

#define HICE_DISPATCH_FLOATING_TYPES(SCALAR_TYPE, NAME, ...)                \
  [&] {                                                                     \
    switch (SCALAR_TYPE) {                                                  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Double, double, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                              \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                \
                        << to_string(SCALAR_TYPE) << "'";                   \
    }                                                                       \
  }()

#define HICE_DISPATCH_COMPLEX_TYPES(SCALAR_TYPE, NAME, ...)     \
  [&] {                                                         \
    switch (SCALAR_TYPE) {                                      \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexFloat,    \
                             std::complex<float>, __VA_ARGS__)  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexDouble,   \
                             std::complex<double>, __VA_ARGS__) \
      default:                                                  \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"    \
                        << to_string(SCALAR_TYPE) << "'";       \
    }                                                           \
  }()

#define HICE_DISPATCH_FLOATING_AND_COMPLEX_TYPES(SCALAR_TYPE, NAME, ...)    \
  [&] {                                                                     \
    switch (SCALAR_TYPE) {                                                  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Float, float, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Double, double, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexFloat,                \
                             std::complex<float>, __VA_ARGS__)              \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexDouble,               \
                             std::complex<double>, __VA_ARGS__)             \
      default:                                                              \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                \
                        << to_string(SCALAR_TYPE) << "'";                   \
    }                                                                       \
  }()

#define HICE_DISPATCH_ALL_TYPES(SCALAR_TYPE, NAME, ...)                       \
  [&] {                                                                       \
    switch (SCALAR_TYPE) {                                                    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt8, uint8_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int8, int8_t, __VA_ARGS__)     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt16, uint16_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int16, int16_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt32, uint32_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int32, int32_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt64, uint64_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int64, int64_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Float, float, __VA_ARGS__)     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Double, double, __VA_ARGS__)   \
      default:                                                                \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                  \
                        << to_string(SCALAR_TYPE) << "'";                     \
    }                                                                         \
  }()

#define HICE_DISPATCH_ALL_TYPES_AND(ADDED_SCALAR_TYPE, SCALAR_TYPE, NAME, ...) \
  [&] {                                                                        \
    switch (SCALAR_TYPE) {                                                     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt8, uint8_t, __VA_ARGS__)    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int8, int8_t, __VA_ARGS__)      \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt16, uint16_t, __VA_ARGS__)  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int16, int16_t, __VA_ARGS__)    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt32, uint32_t, __VA_ARGS__)  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int32, int32_t, __VA_ARGS__)    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt64, uint64_t, __VA_ARGS__)  \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int64, int64_t, __VA_ARGS__)    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Float, float, __VA_ARGS__)      \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Double, double, __VA_ARGS__)    \
      HICE_PRIVATE_CASE_TYPE(                                                  \
          ADDED_SCALAR_TYPE,                                                   \
          decltype(ScalarTypeToCType<ADDED_SCALAR_TYPE>::t), __VA_ARGS__)   \
      default:                                                                 \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                   \
                        << to_string(SCALAR_TYPE) << "'";                      \
    }                                                                          \
  }()

#define HICE_DISPATCH_ALL_AND_COMPLEX_TYPES(SCALAR_TYPE, NAME, ...)           \
  [&] {                                                                       \
    switch (SCALAR_TYPE) {                                                    \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt8, uint8_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int8, int8_t, __VA_ARGS__)     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt16, uint16_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int16, int16_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt32, uint32_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int32, int32_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::UInt64, uint64_t, __VA_ARGS__) \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Int64, int64_t, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Float, float, __VA_ARGS__)     \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Double, double, __VA_ARGS__)   \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::Bool, bool, __VA_ARGS__)       \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexFloat,                  \
                             std::complex<float>, __VA_ARGS__)                \
      HICE_PRIVATE_CASE_TYPE(hice::ScalarType::ComplexDouble,                 \
                             std::complex<double>, __VA_ARGS__)               \
      default:                                                                \
        HICE_LOG(ERROR) << #NAME << " not implemented for '"                  \
                        << to_string(SCALAR_TYPE) << "'";                     \
    }                                                                         \
  }()

} // namespace hice
