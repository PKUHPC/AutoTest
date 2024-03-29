#pragma once
#ifdef AITISA_API_GENERATE_FIGURE
#include <Python.h>
#endif
#ifdef AITISA_API_PYTORCH
#include <torch/torch.h>
#endif

#include <iostream>
#include <libconfig.h++>
#include "gtest/gtest.h"
extern "C" {
#include "src/core/tensor.h"
}
#define CONFIG_FILE "../../config/test/test_data.cfg"

namespace aitisa_api {
static constexpr int loop = 100;
static constexpr int warmup = 5;

extern const DataType aitisa_dtypes[];
extern const Device aitisa_devices[];
inline DataType aitisa_int_to_dtype(int n) {
  return aitisa_dtypes[n];
}
inline Device aitisa_int_to_device(int n) {
  return aitisa_devices[n];
}
inline int aitisa_dtype_to_int(DataType dtype) {
  return static_cast<int>(dtype.code);
}
inline int aitisa_device_to_int(Device device) {
  return static_cast<int>(device.type);
}
inline unsigned int elem_size(int dtype) {
  return static_cast<unsigned int>(aitisa_dtypes[dtype].size);
}

void natural_assign(void* data, unsigned int len, int dtype);
void random_assign(void* data, unsigned int len, int dtype);
void full_float(Tensor t, const float value);

using AITISA_Tensor = Tensor;
using AITISA_Device = Device;
using AITISA_DataType = DataType;

#ifdef AITISA_API_PYTORCH
using time_map_value = std::tuple<double, double, double>;
using time_map = std::map<std::string, time_map_value>;
#else
using time_map_value = std::tuple<double, double>;
using time_map = std::map<std::string, time_map_value>;
#endif

void draw_fig_fun(const time_map& m, const std::string& filename);
#define GREEN "\033[32m"
#define RESET "\033[0m"

#define REGISTER_BASIC(TENSOR, DATA_TYPE, INT_TO_DTYPE, DTYPE_TO_INT, DEVICE, \
                       INT_TO_DEVICE, DEVICE_TO_INT, CREATE, RESOLVE)         \
  class Basic {                                                               \
   public:                                                                    \
    using UserTensor = TENSOR;                                                \
    using UserDataType = DATA_TYPE;                                           \
    using UserDevice = DEVICE;                                                \
    static UserDataType user_int_to_dtype(int data_type_num) {                \
      return INT_TO_DTYPE(data_type_num);                                     \
    }                                                                         \
    static UserDevice user_int_to_device(int device_type_num) {               \
      return INT_TO_DEVICE(device_type_num);                                  \
    }                                                                         \
    static int user_dtype_to_int(UserDataType dtype) {                        \
      return DTYPE_TO_INT(dtype);                                             \
    }                                                                         \
    static int user_device_to_int(UserDevice device) {                        \
      return DEVICE_TO_INT(device);                                           \
    }                                                                         \
    static void user_create(UserDataType dtype, UserDevice device,            \
                            int64_t* dims, int64_t ndim, void* data,          \
                            unsigned int len, UserTensor* tensor) {           \
      CREATE(dtype, device, dims, ndim, data, len, tensor);                   \
    }                                                                         \
    static void user_resolve(UserTensor tensor, UserDataType* dtype,          \
                             UserDevice* device, int64_t** dims,              \
                             int64_t* ndim, void** data, unsigned int* len) { \
      RESOLVE(tensor, dtype, device, dims, ndim, data, len);                  \
    }                                                                         \
  };

// functions for debug
template <typename T>
void print_data(T* data, unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void print_data2d(T* data, unsigned int m, unsigned int n) {
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      std::cout << data[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
}

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
  static const size_t nargs = sizeof...(Args);

  typedef R result_type;

  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

template <size_t I, typename T>
struct type_traits;

template <size_t I, typename R, typename... Args>
struct type_traits<I, std::function<R(Args...)>> {
  typedef typename std::tuple_element<I, std::tuple<Args...>>::type type;
};

template <std::size_t N, typename F, typename U>
struct TypeComparer {
  static void compare() {
    TypeComparer<N - 1, F, U>::compare();
    typedef typename type_traits<N - 1, F>::type type_f;
    typedef typename type_traits<N - 1, U>::type type_u;
    if (!std::is_same<typename std::remove_cv<type_f>::type, type_u>::value) {
      throw std::invalid_argument("Incorrect type: type mismatch at argument " +
                                  std::to_string(N - 1));
    }
  }
};

template <typename F, typename U>
struct TypeComparer<1, F, U> {
  static void compare() {
    typedef typename type_traits<0, F>::type type_f;
    typedef typename type_traits<0, U>::type type_u;
    if (!std::is_same<typename std::remove_cv<type_f>::type, type_u>::value) {
      throw std::invalid_argument("Incorrect type: type mismatch at argument " +
                                  std::to_string(0));
    }
  }
};

template <size_t N, typename F, typename U>
struct TypeCompare;

template <size_t N, typename F, typename U>
struct TypeCompare {
  TypeCompare() { TypeComparer<N, F, U>::compare(); }
};

}  // namespace aitisa_api
#ifdef AITISA_API_PYTORCH
namespace libtorch_api {

using Tensor = torch::Tensor;
using DataType = c10::ScalarType;
using Device = c10::Device;
const DataType torch_dtypes[10] = {
    torch::kInt8,  torch::kUInt8, torch::kInt16, torch::kInt16, torch::kInt32,
    torch::kInt32, torch::kInt64, torch::kInt64, torch::kFloat, torch::kDouble,
};

static std::map<const char*, int> torchTypeMap{
    {"Char", 0}, {"Byte", 1},  {"Short", 2}, {"Int", 4},
    {"Long", 6}, {"Float", 8}, {"Double", 9}};
inline int torch_dtype_to_int(DataType dtype) {
  return torchTypeMap[c10::toString(dtype)];
}
inline DataType torch_int_to_dtype(int n) {
  return torch_dtypes[n];
}
inline Device torch_int_to_device(int n) {
  return c10::Device(c10::DeviceType::CPU);
}
inline int torch_device_to_int(Device device) {
  return static_cast<int>(device.type());
}
inline void torch_create(DataType dtype, Device device, int64_t* dims,
                         int64_t ndim, void* data, unsigned int len,
                         Tensor* output) {

  at::IntArrayRef array(dims, ndim);
  torch::Tensor tensor =
      torch::empty(array, torch::TensorOptions().device(device).dtype(dtype));
  torch::DataPtr DataPtr(data, device);
  tensor.storage().set_data_ptr(std::move(DataPtr));

  *output = tensor;
}
inline void torch_resolve(const Tensor& input, DataType* dtype, Device& device,
                          int64_t** dims, int64_t* ndim, void** data,
                          unsigned int* len) {
  *dtype = input.scalar_type();
  device = input.device();
  torch::IntArrayRef array = input.sizes();
  *dims = const_cast<int64_t*>(array.data());
  *ndim = input.dim();
  void* data_ = const_cast<void*>(input.data_ptr());
  *data = data_;
  *len = input.nbytes();
}

}  // namespace libtorch_api
#endif