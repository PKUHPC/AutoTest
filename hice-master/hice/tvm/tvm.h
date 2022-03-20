#pragma once

#include <hice/core/tensor.h>
#include <hice/util/dlpack.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/te/operation.h>
#include <tvm/driver/driver_api.h>
 
#include <mutex>
#include <memory>

using namespace tvm;
using namespace tvm::te;

namespace hice {

class TVMHandle;
typedef std::shared_ptr<TVMHandle> TVMHandlePtr;

class HICE_API TVMLibConfig {
public:
  static std::string name() { return name_; }
  static std::string prefix() { return prefix_; }
  static int n_search_trails() { return n_search_trails_; }
  
  static void set_name(const std::string& name) { name_ = name; }
  static void set_prefix(const std::string& prefix) { prefix_ = prefix + "/"; }
  static int set_n_search_trails(int n_search_trails) { n_search_trails_ = n_search_trails; }

private:
  TVMLibConfig() {};
  static std::string name_;
  static std::string prefix_;
  static int n_search_trails_;
};

class TVMHandle {
  // use shared_ptr to avoid memory leak
  typedef std::shared_ptr<runtime::Module> ModulePtr;

public:
  void clear_module();
  runtime::PackedFunc get(const std::string& func_name);
  static TVMHandlePtr getInstance();

private:
  std::mutex mu_;
  ModulePtr rt_module_;

  TVMHandle(): mu_(), rt_module_(nullptr)  { }
  TVMHandle(TVMHandle const&) = delete;
  TVMHandle& operator=(TVMHandle const&) = delete;
};  // class TVMHandle

std::vector<std::string> getFileList(std::string path, std::string format);

HICE_API inline bool is_tvm_available() {
#ifdef HICE_USE_TVM
  return true;
#else
  return false;
#endif
}

} // namespace hice
