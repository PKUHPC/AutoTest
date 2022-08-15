#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/core/tensor_printer.h"

extern "C" {
#include "src/nn/dropout.h"
}

namespace aitisa_api {

namespace {

class Dropout_Input : public Unary_Input {
 public:
  Dropout_Input() = default;
  ;
  Dropout_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
                void* data, unsigned int len, double rate, int initvalue)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len),
        rate_(rate),
        initvalue_(initvalue) {}
  Dropout_Input(Dropout_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        rate_(input.rate()),
        initvalue_(input.initvalue()) {
    input.to_nullptr();
  }
  ~Dropout_Input() override = default;

  Dropout_Input& operator=(Dropout_Input& right) {
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->rate_ = right.rate();
    this->initvalue_ = right.initvalue();
  }
  double rate() const { return rate_; }
  int initvalue() const { return initvalue_; }

 private:
  double rate_ = 0;
  int initvalue_ = 0;
};
}  // namespace

template <typename InterfaceType>
class DropoutTest : public ::testing::Test {
 public:
  DropoutTest() { fetch_test_data("drop_out", drop_out_inputs, drop_out_name); }
  ~DropoutTest() override = default;
  static void aitisa_kernel(const AITISA_Tensor input, const double rate,
                            AITISA_Tensor* output) {
    aitisa_dropout(input, rate, output);
  }
  int fetch_test_data(const char* path, std::vector<Dropout_Input>& inputs,
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
        int test_index, ndim, dtype, device, len, initvalue;
        std::string input_name;
        float rate;

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
        if (!setting.lookupValue("initvalue", initvalue)) {
          std::cerr << "Setting \"initvalue\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("rate", rate)) {
          std::cerr << "Setting \"rate\" do not exist in test index "
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
        Dropout_Input tmp(ndim, dims, dtype, device, nullptr, len, rate,
                          initvalue);
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
  using InputType = Dropout_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Dropout_Input> drop_out_inputs;
  std::vector<std::string> drop_out_name;
  std::map<std::string, int> test_case = {{"drop_out", 0}};
};
TYPED_TEST_CASE_P(DropoutTest);

TYPED_TEST_P(DropoutTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  time_map m;
  auto test = [&m](std::vector<Dropout_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      int64_t aitisa_result_ndim, user_result_ndim;
      int64_t *aitisa_result_dims = nullptr, *user_result_dims = nullptr;
      float *aitisa_result_data = nullptr, *user_result_data = nullptr;
      unsigned int aitisa_result_len, user_result_len;
      AITISA_Tensor aitisa_tensor, aitisa_result;
      AITISA_DataType aitisa_result_dtype;
      AITISA_Device aitisa_result_device;
      UserTensor user_tensor, user_result;
      UserDataType user_result_dtype;
      UserDevice user_result_device;
      // aitisa
      AITISA_DataType aitisa_dtype = aitisa_int_to_dtype(inputs[i].dtype());
      AITISA_Device aitisa_device =
          aitisa_int_to_device(0);  // cpu supoorted only
      aitisa_create(aitisa_dtype, aitisa_device, inputs[i].dims(),
                    inputs[i].ndim(), (void*)(inputs[i].data()),
                    inputs[i].len(), &aitisa_tensor);
      full_float(aitisa_tensor, inputs[i].initvalue());

      auto aitisa_start = std::chrono::steady_clock::now();

      aitisa_dropout(aitisa_tensor, inputs[i].rate(), &aitisa_result);

      auto aitisa_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> aitisa_elapsed = aitisa_end - aitisa_start;
      aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device,
                     &aitisa_result_dims, &aitisa_result_ndim,
                     (void**)&aitisa_result_data, &aitisa_result_len);

      // user
      UserDataType user_dtype = UserFuncs::user_int_to_dtype(inputs[i].dtype());
      UserDevice user_device =
          UserFuncs::user_int_to_device(inputs[i].device());
      UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                             inputs[i].ndim(), inputs[i].data(),
                             inputs[i].len(), &user_tensor);

      auto user_start = std::chrono::steady_clock::now();
      UserFuncs::user_dropout(user_tensor, inputs[i].rate(), &user_result);
      auto user_end = std::chrono::steady_clock::now();

      std::chrono::duration<double> user_elapsed = user_end - user_start;

      UserFuncs::user_resolve(user_result, &user_result_dtype,
                              &user_result_device, &user_result_dims,
                              &user_result_ndim, (void**)&user_result_data,
                              &user_result_len);

      // compare
      int64_t tensor_size = 1;
      ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
      ASSERT_EQ(/*CUDA*/ 0, UserFuncs::user_device_to_int(user_result_device));
      ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                UserFuncs::user_dtype_to_int(user_result_dtype));
      for (int64_t j = 0; j < aitisa_result_ndim; j++) {
        tensor_size *= aitisa_result_dims[j];
        ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
      }
      ASSERT_EQ(aitisa_result_len, user_result_len);

      auto* aitisa_data = (float*)aitisa_result_data;
      auto* user_data = (float*)user_result_data;
      int aitisa_count = 0;
      int user_count = 0;
      for (int64_t j = 0; j < tensor_size; j++) {
        if (aitisa_data[j] == 0) {
          aitisa_count++;
        }
        if (user_data[j] == 0) {
          user_count++;
        }
      }
      ASSERT_TRUE(abs((double)aitisa_count / (double)tensor_size -
                      (double)user_count / (double)tensor_size) < 1e-3);

      // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i
                << " / " << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] "
                << /*RESET <<*/ aitisa_elapsed.count() * 1000 << " ms"
                << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] "
                << /*RESET <<*/ user_elapsed.count() * 1000 << " ms"
                << std::endl;
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value(aitisa_elapsed.count() * 1000,
                                             user_elapsed.count() * 1000)));
    }
  };
  if (this->drop_out_inputs.size()) {
    test(std::move(this->drop_out_inputs), std::move(this->drop_out_name),
         "drop_out", this->test_case["drop_out"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "drop_out");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(DropoutTest, TwoTests);

#define REGISTER_DROPOUT(DROPOUT_FUNC, DROPOUT)                                \
  class Dropout : public Basic {                                               \
   public:                                                                     \
    static void user_dropout(UserTensor input, const double rate,              \
                             UserTensor* output) {                             \
      typedef std::function<void(UserTensor, double, UserTensor*)>             \
          dropout_func;                                                        \
      auto func_args_num = aitisa_api::function_traits<DROPOUT_FUNC>::nargs;   \
      auto args_num = aitisa_api::function_traits<dropout_func>::nargs;        \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<std::remove_cv<aitisa_api::function_traits<            \
                            dropout_func>::result_type>::type,                 \
                        aitisa_api::function_traits<                           \
                            DROPOUT_FUNC>::result_type>::value) {              \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<                                                 \
          aitisa_api::function_traits<dropout_func>::nargs, dropout_func,      \
          DROPOUT_FUNC>();                                                     \
      DROPOUT(input, rate, output);                                            \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, DropoutTest, Dropout);             \
  }

}  // namespace aitisa_api
