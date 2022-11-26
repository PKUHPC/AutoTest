#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/core/tensor_printer.h"

namespace aitisa_api {

namespace {
hice::TensorPrinter tp;

class CtcLossInput : public Unary_Input {
 public:
  CtcLossInput() = default;

  CtcLossInput(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
               void* data, unsigned int len, int batch_size, int max_time,
               int max_length, int n_classes, int reduction)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len),
        batch_size_(batch_size),
        max_time_(max_time),
        max_length_(max_length),
        n_classes_(n_classes),
        reduction_(reduction) {}
  CtcLossInput(CtcLossInput&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        batch_size_(input.batch_size()),
        max_time_(input.max_time()),
        max_length_(input.max_length()),
        n_classes_(input.n_classes()),
        reduction_(input.reduction()) {
    input.to_nullptr();
  }
  ~CtcLossInput() override = default;

  CtcLossInput& operator=(CtcLossInput& right) {
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->batch_size_ = right.batch_size();
    this->max_time_ = right.max_time();
    this->max_length_ = right.max_length();
    this->n_classes_ = right.n_classes();
    this->reduction_ = right.reduction();
  }
  int batch_size() const { return batch_size_; }
  int max_time() const { return max_time_; }
  int max_length() const { return max_length_; }
  int n_classes() const { return n_classes_; }
  int reduction() const { return reduction_; }
  void ctc_loss_set_data(void* prods, void* target, void* probs_lengths,
                         void* target_lengths) {
    this->prods_ = prods;
    this->target_ = target;
    this->probs_lengths_ = probs_lengths;
    this->target_lengths_ = target_lengths;
  }
  void* prods() const { return prods_; };
  void* target() const { return target_; };
  void* probs_lengths() const { return probs_lengths_; };
  void* target_lengths() const { return target_lengths_; };

 private:
  int batch_size_ = 0;
  int max_time_ = 0;
  int max_length_ = 0;
  int n_classes_ = 0;
  int reduction_ = 0;

  void* prods_ = nullptr;
  void* target_ = nullptr;
  void* probs_lengths_ = nullptr;
  void* target_lengths_ = nullptr;
};
}  // namespace

template <typename InterfaceType>
class CtcLossTest : public ::testing::Test {
 public:
  CtcLossTest() {
    fetch_test_data("ctc_loss", ctc_loss_inputs, ctc_loss_inputs_name);
  }
  ~CtcLossTest() override = default;
  int fetch_test_data(const char* path, std::vector<CtcLossInput>& inputs,
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
        int test_index, ndim, dtype, device, len, batch_size, max_time,
            max_length, n_classes, reduction;
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
        if (!setting.lookupValue("batch_size", batch_size)) {
          std::cerr << "Setting \"batch_size\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("max_time", max_time)) {
          std::cerr << "Setting \"max_time\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("max_length", max_length)) {
          std::cerr << "Setting \"max_length\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("n_classes", n_classes)) {
          std::cerr << "Setting \"n_classes\" do not exist in test index "
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
        if (!setting.lookupValue("reduction", reduction)) {
          std::cerr << "Setting \"reduction\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        CtcLossInput tmp(ndim, dims, dtype, device, nullptr, len, batch_size,
                         max_time, max_length, n_classes, reduction);
        inputs.push_back(std::move(tmp));
        inputs_name.push_back(input_name);
      }

    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
  }
  using InputType = CtcLossInput;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<CtcLossInput> ctc_loss_inputs;
  std::vector<std::string> ctc_loss_inputs_name;
  std::map<std::string, int> test_case = {{"ctc_loss", 0}};
};
TYPED_TEST_CASE_P(CtcLossTest);

TYPED_TEST_P(CtcLossTest, TwoTests) {
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
  auto test = [&m](std::vector<CtcLossInput>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto user_elapsed = std::chrono::duration<double>::zero();
#ifdef AITISA_API_PYTORCH
      auto torch_elapsed = std::chrono::duration<double>::zero();
#endif
      //loop test
      for (int n = 0; n < 1; n++) {
        int64_t user_result_ndim;
        int64_t* user_result_dims = nullptr;
        void* user_result_data = nullptr;
        unsigned int user_result_len;
        UserTensor user_tensor, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        void* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_prods_tensor, torch_target_tensor,
            torch_probs_lengths_tensor, torch_target_lengths_tensor,
            torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
        // user
        UserDataType user_dtype =
            UserFuncs::user_int_to_dtype(inputs[i].dtype());
        UserDevice user_device =
            UserFuncs::user_int_to_device(inputs[i].device());

        UserTensor probs = hice::rand_uniform(
            {inputs[i].max_time(), inputs[i].batch_size(),
             inputs[i].n_classes()},
            0.0, 1.0, hice::device(user_device).dtype(user_dtype));
        UserTensor target =
            hice::rand_uniform({inputs[i].batch_size(), inputs[i].max_length()},
                               1, inputs[i].n_classes(),
                               hice::device(user_device).dtype(hice::kInt32));
        UserTensor probs_lengths =
            hice::full({inputs[i].batch_size()}, inputs[i].max_time(),
                       hice::device(user_device).dtype(hice::kInt32));
        UserTensor target_lengths = hice::rand_uniform(
            {inputs[i].batch_size()}, 0, inputs[i].max_length(),
            hice::device(user_device).dtype(hice::kInt32));
        inputs[i].ctc_loss_set_data(
            const_cast<void*>(probs.raw_data()),
            const_cast<void*>(target.raw_data()),
            const_cast<void*>(probs_lengths.raw_data()),
            const_cast<void*>(target_lengths.raw_data()));

        auto user_start = std::chrono::steady_clock::now();
        tp.print(probs);
        tp.print(target);
        tp.print(probs_lengths);

        tp.print(target_lengths);
        UserTensor input;
        UserFuncs::user_softmax(probs, 2, &input);
        tp.print(input);

        user_result = std::get<0>(
            UserFuncs::user_ctc_loss(input, target, probs_lengths,
                                     target_lengths, inputs[i].reduction()));

        auto user_end = std::chrono::steady_clock::now();
        tp.print(user_result);

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

        int64_t* prods_dims = new int64_t[3];

        prods_dims[0] = inputs[i].max_time();
        prods_dims[1] = inputs[i].batch_size();
        prods_dims[2] = inputs[i].n_classes();
        libtorch_api::torch_create(torch_dtype, torch_device, prods_dims, 3,
                                   inputs[i].prods(), inputs[i].len(),
                                   &torch_prods_tensor);

        int64_t* target_dims = new int64_t[2];
        target_dims[0] = inputs[i].batch_size();
        target_dims[1] = inputs[i].max_length();
        libtorch_api::torch_create(torch::kInt32, torch_device, target_dims, 2,
                                   inputs[i].target(), inputs[i].len(),
                                   &torch_target_tensor);

        int64_t* prods_length_dims = new int64_t[1];
        prods_length_dims[0] = inputs[i].batch_size();
        libtorch_api::torch_create(torch::kInt32, torch_device,
                                   prods_length_dims, 1,
                                   inputs[i].probs_lengths(), inputs[i].len(),
                                   &torch_probs_lengths_tensor);

        int64_t* target_length_dims = new int64_t[1];
        target_length_dims[0] = inputs[i].batch_size();
        libtorch_api::torch_create(torch::kInt32, torch_device,
                                   target_length_dims, 1,
                                   inputs[i].target_lengths(), inputs[i].len(),
                                   &torch_target_lengths_tensor);

        auto torch_start = std::chrono::steady_clock::now();
        std::cout << "torch_prods_tensor" << std::endl
                  << torch_prods_tensor << std::endl;
        std::cout << "torch_target_tensor" << std::endl
                  << torch_target_tensor << std::endl;
        std::cout << "torch_probs_lengths_tensor" << std::endl
                  << torch_probs_lengths_tensor << std::endl;
        std::cout << "torch_target_lengths_tensor" << std::endl
                  << torch_target_lengths_tensor << std::endl;

        TorchTensor torch_input = torch::log_softmax(torch_prods_tensor, 2);
        torch_result = torch::ctc_loss(
            torch_input, torch_target_tensor, torch_probs_lengths_tensor,
            torch_target_lengths_tensor, 0, at::Reduction::None);
        //            (inputs[i].reduction() == 1 ? at::Reduction::Mean
        //                                        : at::Reduction::Sum));

        std::cout << torch_result << std::endl;
        auto torch_end = std::chrono::steady_clock::now();
        torch_elapsed += torch_end - torch_start;
        libtorch_api::torch_resolve(
            torch_result, &torch_result_dtype, torch_result_device,
            &torch_result_dims, &torch_result_ndim, (void**)&torch_result_data,
            &torch_result_len);
#endif
// compare
#ifdef AITISA_API_PYTORCH
        int64_t tensor_size = 1;
        ASSERT_EQ(user_result_ndim, torch_result_ndim);
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
#ifdef AITISA_API_PYTORCH
        auto* torch_data = (float*)torch_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          if (inputs[i].reduction() == 2) {
            ASSERT_TRUE(abs(user_data[j] - torch_data[j]) < 1e-1);
          } else {
            ASSERT_TRUE(abs(user_data[j] - torch_data[j]) < 1e-2);
          }
        }
#endif
        delete[] prods_dims;
        delete[] target_dims;
        delete[] prods_length_dims;
        delete[] target_length_dims;
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
  if (this->ctc_loss_inputs.size()) {
    test(std::move(this->ctc_loss_inputs),
         std::move(this->ctc_loss_inputs_name), "ctc_loss",
         this->test_case["ctc_loss"]);
#ifdef AITISA_API_GENERATE_FIGURE
//    draw_fig_fun(m, "drop_out");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(CtcLossTest, TwoTests);

#define REGISTER_CTCLOSS(CTCLOSS, SOFTMAX)                            \
  class CtcLoss : public Basic {                                      \
   public:                                                            \
    static std::tuple<UserTensor, UserTensor> user_ctc_loss(          \
        UserTensor prods, UserTensor traget, UserTensor prods_length, \
        UserTensor target_length, const int reduction) {              \
      return CTCLOSS(prods, traget, prods_length, target_length,      \
                     hice::Reduction::none);                          \
    }                                                                 \
    static void user_softmax(UserTensor input, const int axis,        \
                             UserTensor* output) {                    \
      SOFTMAX(input, axis, output);                                   \
    }                                                                 \
  };                                                                  \
  namespace aitisa_api {                                              \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, CtcLossTest, CtcLoss);    \
  }

}  // namespace aitisa_api
