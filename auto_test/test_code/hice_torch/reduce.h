#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

namespace aitisa_api {

namespace {

class Reduce_Input : public Unary_Input {
 public:
  Reduce_Input() = default;
  Reduce_Input(int64_t ndim, int64_t* dims, int dtype, int device, void* data,
               unsigned int len, int* dim, int dim_len, bool keepdim)
      : Unary_Input(ndim, dims, dtype, device, data, len),
        dim_(dim),
        dim_len_(dim_len),
        keepdim_(keepdim == 1) {}
  Reduce_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
               void* data, unsigned int len, std::vector<int> dim, bool keepdim)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len),
        dim_(nullptr),
        dim_len_(dim.size()),
        keepdim_(keepdim == 1) {
    this->dim_ = new int[dim_len_];
    for (int i = 0; i < dim_len_; i++) {
      this->dim_[i] = dim[i];
    }
  }
  Reduce_Input(Reduce_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        dim_(input.dim()),
        dim_len_(input.dim_len()),
        keepdim_(keepdim()) {
    input.to_nullptr();
    input.dim_ = nullptr;
  }
  ~Reduce_Input() override { delete[] dim_; }
  Reduce_Input& operator=(Reduce_Input const& right) {
    int spatial_len = right.dim_len();
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->dim_ = new int[spatial_len];
    memcpy(this->dim_, right.dim(), spatial_len * sizeof(int));
  }
  int* dim() const { return dim_; }
  int dim_len() const { return dim_len_; }
  bool keepdim() const { return keepdim_; }

 private:
  int* dim_ = nullptr;
  int dim_len_ = 0;
  bool keepdim_ = false;
};

}  // namespace

template <typename InterfaceType>
class ReduceTest : public ::testing::Test {
 public:
  ReduceTest() {
    fetch_test_data("reduce.sum", reduce_sum_inputs,
                    reduce_sum_inputs_name);
    fetch_test_data("reduce.mean", reduce_mean_inputs,
                    reduce_mean_inputs_name);
    fetch_test_data("reduce.min", reduce_min_inputs,
                    reduce_min_inputs_name);
    fetch_test_data("reduce.max", reduce_max_inputs,
                    reduce_max_inputs_name);
  }
  ~ReduceTest() override = default;
  using InputType = Reduce_Input;
  using UserInterface = InterfaceType;

  int fetch_test_data(const char* path, std::vector<Reduce_Input>& inputs,
                      std::vector<std::string>& inputs_name) {

    libconfig::Config cfg;

    try {
      cfg.readFile(CONFIG_FILE);
    } catch (const libconfig::FileIOException& fioex) {
      std::cerr << "I/O error while reading file." << std::endl;
      return (EXIT_FAILURE);
    } catch (const libconfig::ParseException& pex) {
      std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                << " - " << pex.getError() << std::endl;
      return (EXIT_FAILURE);
    }

    try {
      const libconfig::Setting& settings = cfg.lookup(path);
      int count = settings.getLength();

      for (int i = 0; i < count; ++i) {
        const libconfig::Setting& setting = settings[i];

        std::vector<int64_t> dims;
        std::vector<int> dim;
        int test_index, ndim, dtype, device, len, keepdim_int;
        std::string input_name;

        if (!setting.lookupValue("test_index", test_index)) {
          std::cerr << "Setting \"test_index\" do not exist in " << path
                    << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("ndim", ndim)) {
          std::cerr << "Setting \"ndim\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dims_setting = setting.lookup("dims");
          if (dims_setting.getLength() != ndim) {
            std::cerr << " \"dims\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < dims_setting.getLength(); ++n) {
            dims.push_back((int64_t) int(dims_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dims\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("dtype", dtype)) {
          std::cerr << "Setting \"dtype\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("device", device)) {
          std::cerr << "Setting \"device\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("len", len)) {
          std::cerr << "Setting \"len\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dim_setting = setting.lookup("dim");
          for (int n = 0; n < dim_setting.getLength(); ++n) {
            dim.push_back(int(dim_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dim\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("input_name", input_name)) {
          std::cerr << "Setting \"input_name\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("keepdim", keepdim_int)) {
          std::cerr << "Setting \"keepdim\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        Reduce_Input tmp(ndim, dims, dtype, device, nullptr, len, dim,
                         keepdim_int);
        inputs.push_back(std::move(tmp));
        inputs_name.push_back(input_name);
      }

      for (auto& input : inputs) {
        unsigned int input_nelem = 1;
        for (unsigned int j = 0; j < input.ndim(); j++) {
          input_nelem *= input.dims()[j];
        }

        unsigned int input_len = input_nelem * elem_size(input.dtype());
        void* input_data = (void*)new char[input_len];
        random_assign(input_data, input_len, input.dtype());
        input.set_data(input_data, input_len);
      }
    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
  }
  // inputs
  std::vector<Reduce_Input> reduce_sum_inputs;
  std::vector<std::string> reduce_sum_inputs_name;

  std::vector<Reduce_Input> reduce_mean_inputs;
  std::vector<std::string> reduce_mean_inputs_name;

  std::vector<Reduce_Input> reduce_min_inputs;
  std::vector<std::string> reduce_min_inputs_name;

  std::vector<Reduce_Input> reduce_max_inputs;
  std::vector<std::string> reduce_max_inputs_name;
  std::map<std::string, int> test_case = {{"reduce_sum", 0},
                                          {"reduce_mean", 1},
                                          {"reduce_min", 2},
                                          {"reduce_max", 3}};
};
TYPED_TEST_CASE_P(ReduceTest);

TYPED_TEST_P(ReduceTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  time_map m;
  auto test = [&m](std::vector<Reduce_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto user_elapsed = std::chrono::duration<double>::zero();
#ifdef AITISA_API_PYTORCH
      auto torch_elapsed = std::chrono::duration<double>::zero();
#endif
      //loop test
      for (int n = 0; n < loop; n++) {
        int64_t user_result_ndim;
        int64_t* user_result_dims = nullptr;
        float* user_result_data = nullptr;
        unsigned int user_result_len;
        UserTensor user_tensor, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        float* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor, torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
        // user
        UserDataType user_dtype =
            UserFuncs::user_int_to_dtype(inputs[i].dtype());
        UserDevice user_device =
            UserFuncs::user_int_to_device(inputs[i].device());
        UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                               inputs[i].ndim(), inputs[i].data(),
                               inputs[i].len(), &user_tensor);
        auto user_start = std::chrono::steady_clock::now();
        switch (test_case_index) {
          case 0:
            user_result = UserFuncs::user_reduce_sum(
                user_tensor, inputs[i].dim(), inputs[i].dim_len(),
                inputs[i].keepdim());
            break;
          case 1:
            user_result = UserFuncs::user_reduce_mean(
                user_tensor, inputs[i].dim(), inputs[i].dim_len(),
                inputs[i].keepdim());
            break;
          case 2:
            user_result = UserFuncs::user_reduce_min(
                user_tensor, inputs[i].dim(), inputs[i].dim_len(),
                inputs[i].keepdim());
            break;
          case 3:
            user_result = UserFuncs::user_reduce_max(
                user_tensor, inputs[i].dim(), inputs[i].dim_len(),
                inputs[i].keepdim());
            break;
          default:
            break;
        }
        auto user_end = std::chrono::steady_clock::now();
        user_elapsed += user_end - user_start;
        UserFuncs::user_resolve(user_result, &user_result_dtype,
                                &user_result_device, &user_result_dims,
                                &user_result_ndim, (void**)&user_result_data,
                                &user_result_len);
#ifdef AITISA_API_PYTORCH
        //torch
        TorchDataType torch_dtype =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype());
        TorchDevice torch_device =
            libtorch_api::torch_int_to_device(inputs[i].device());
        libtorch_api::torch_create(torch_dtype, torch_device, inputs[i].dims(),
                                   inputs[i].ndim(), inputs[i].data(),
                                   inputs[i].len(), &torch_tensor);

        std::vector<int64_t> dim_list;

        for (int index = 0; index < inputs[i].dim_len(); index++) {
          dim_list.push_back(inputs[i].dim()[index]);
        }
        auto torch_start = std::chrono::steady_clock::now();

        switch (test_case_index) {
          case 0:
            torch_result =
                torch::sum(torch_tensor, dim_list, inputs[i].keepdim());
            break;
          case 1:
            torch_result =
                torch::mean(torch_tensor, dim_list, inputs[i].keepdim());
            break;
          case 2:
            torch_result = std::get<0>(
                torch::min(torch_tensor, dim_list[0], inputs[i].keepdim()));
            break;
          case 3:
            torch_result = std::get<0>(
                torch::max(torch_tensor, dim_list[0], inputs[i].keepdim()));
            break;
          default:
            break;
        }

        auto torch_end = std::chrono::steady_clock::now();
        torch_elapsed += torch_end - torch_start;
        libtorch_api::torch_resolve(
            torch_result, &torch_result_dtype, torch_result_device,
            &torch_result_dims, &torch_result_ndim, (void**)&torch_result_data,
            &torch_result_len);
#endif
        // compare
#ifdef AITISA_API_PYTORCH
        ASSERT_EQ(user_result_ndim, torch_result_ndim);
        int64_t tensor_size = 1;
        ASSERT_EQ(UserFuncs::user_device_to_int(user_result_device),
                  libtorch_api::torch_device_to_int(torch_result_device));
        ASSERT_EQ(UserFuncs::user_dtype_to_int(user_result_dtype),
                  libtorch_api::torch_dtype_to_int(torch_result_dtype));
        for (int64_t j = 0; j < user_result_ndim; j++) {
          tensor_size *= user_result_dims[j];
          ASSERT_EQ(user_result_dims[j], torch_result_dims[j]);
        }
        ASSERT_EQ(user_result_len, torch_result_len);
#endif

        auto* user_data = (float*)user_result_data;
        auto* torch_data = (float*)torch_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          if (test_case_index == 2 || test_case_index == 3) {
            ASSERT_EQ(user_data[j], torch_data[j]);
          } else {
            ASSERT_TRUE(abs(user_data[j] - torch_data[j]) < 1e-2);
          }
        }
      }
      auto user_time = user_elapsed.count() * 1000 / loop;

      // print result of test
      std::cout << "[ " << test_case_name << " sample" << i << " / "
                << inputs_name[i] << " ] " << std::endl;
      std::cout << "\t[  USER  ] " << user_time << " ms average for " << loop
                << " loop " << std::endl;
#ifdef AITISA_API_PYTORCH
      auto torch_time = torch_elapsed.count() * 1000 / loop;
      std::cout << "\t[  TORCH  ] " << torch_time << " ms average for " << loop
                << " loop " << std::endl;
//      m.insert(
//          std::make_pair(test_case_name + " sample " + std::to_string(i),
//                         time_map_value(aitisa_time, user_time, torch_time)));
#else
//      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
//                              time_map_value(aitisa_time, user_time)));
#endif
    }
  };
  if (this->reduce_sum_inputs.size() && this->reduce_mean_inputs.size() &&
      this->reduce_min_inputs.size() && this->reduce_max_inputs.size()) {
    test(std::move(this->reduce_sum_inputs),
         std::move(this->reduce_sum_inputs_name), "reduce_sum",
         this->test_case["reduce_sum"]);
    test(std::move(this->reduce_mean_inputs),
         std::move(this->reduce_mean_inputs_name), "reduce_mean",
         this->test_case["reduce_mean"]);
    test(std::move(this->reduce_min_inputs),
         std::move(this->reduce_min_inputs_name), "reduce_min",
         this->test_case["reduce_min"]);
    test(std::move(this->reduce_max_inputs),
         std::move(this->reduce_max_inputs_name), "reduce_max",
         this->test_case["reduce_max"]);
#ifdef AITISA_API_GENERATE_FIGURE
//    draw_fig_fun(m, "pooling");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(ReduceTest, TwoTests);

#define REGISTER_REDUCE(REDUCE_SUM, REDUCE_MEAN, REDUCE_MIN, REDUCE_MAX)       \
  class Reduce : public Basic {                                                \
   public:                                                                     \
    static UserTensor user_reduce_sum(UserTensor input, const int* dim,        \
                                      const int dim_len, const bool keepdim) { \
      std::vector<int64_t> dim_vector = {};                                    \
      dim_vector.reserve(dim_len);                                             \
      for (auto i = 0; i < dim_len; i++) {                                     \
        dim_vector.push_back(dim[i]);                                          \
      }                                                                        \
      return REDUCE_SUM(input, dim_vector, keepdim);                           \
    }                                                                          \
    static UserTensor user_reduce_mean(UserTensor input, const int* dim,       \
                                       const int dim_len,                      \
                                       const bool keepdim) {                   \
      std::vector<int64_t> dim_vector = {};                                    \
      dim_vector.reserve(dim_len);                                             \
      for (auto i = 0; i < dim_len; i++) {                                     \
        dim_vector.push_back(dim[i]);                                          \
      }                                                                        \
      return REDUCE_MEAN(input, dim_vector, keepdim);                          \
    }                                                                          \
    static UserTensor user_reduce_min(UserTensor input, const int* dim,        \
                                      const int dim_len, const bool keepdim) { \
      std::vector<int64_t> dim_vector = {};                                    \
      dim_vector.reserve(dim_len);                                             \
      for (auto i = 0; i < dim_len; i++) {                                     \
        dim_vector.push_back(dim[i]);                                          \
      }                                                                        \
      return REDUCE_MIN(input, dim_vector, keepdim);                           \
    }                                                                          \
    static UserTensor user_reduce_max(UserTensor input, const int* dim,        \
                                      const int dim_len, const bool keepdim) { \
      std::vector<int64_t> dim_vector = {};                                    \
      dim_vector.reserve(dim_len);                                             \
      for (auto i = 0; i < dim_len; i++) {                                     \
        dim_vector.push_back(dim[i]);                                          \
      }                                                                        \
      return REDUCE_MAX(input, dim_vector, keepdim);                           \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, ReduceTest, Reduce);               \
  }

}  // namespace aitisa_api