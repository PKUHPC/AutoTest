#include <hice/tvm/tvm.h>
#include <hice/util/dlpack_wrapper.h>

#include <string>
#include <vector>
#include <thread>

#include <cstring>
#include <dirent.h>

using namespace tvm;
using namespace tvm::te;

namespace hice {

std::string TVMLibConfig::name_ = "libhice_tvm_ops.so";
std::string TVMLibConfig::prefix_ = "./";
int TVMLibConfig::n_search_trails_ = 5;

void TVMHandle::clear_module() {
  if (rt_module_) {
    std::lock_guard<std::mutex> lock(mu_);
    if (rt_module_) {
      rt_module_.reset();
    }
  }
}

runtime::PackedFunc TVMHandle::get(const std::string& func_name) {
  if (rt_module_) return rt_module_->GetFunction(func_name);
  std::lock_guard<std::mutex> lock(mu_);
  if (rt_module_) return rt_module_->GetFunction(func_name);
  // check dir
  std::string rtm_path_ = TVMLibConfig::prefix() + TVMLibConfig::name();
  // complie
  HICE_DLOG(INFO) << "compling tvm module.";
  std::stringstream s;
  s << "gcc -shared -fPIC -o " << rtm_path_ << " ";
  auto ojb_files = getFileList(TVMLibConfig::prefix(), ".o");
  if (ojb_files.size() == 0) return nullptr;
  for (auto& obj : ojb_files) {
    s << obj << " ";
  }
  auto ret = system(s.str().c_str());
  HICE_CHECK_EQ(ret, 0);
  HICE_DLOG(INFO) << "loading tvm module.";
  // load
  rt_module_ = std::make_shared<runtime::Module>(runtime::Module::LoadFromFile(rtm_path_, "so"));
  auto ptx_files = getFileList(TVMLibConfig::prefix(), ".ptx");
  for (auto& ptx : ptx_files) {
    rt_module_->Import(runtime::Module::LoadFromFile(ptx));
  }
  return rt_module_->GetFunction(func_name);
}

TVMHandlePtr TVMHandle::getInstance() {
  static TVMHandlePtr instance(new TVMHandle());
  return instance;
}
 
std::vector<std::string> getFileList(std::string path, std::string format) {
  std::vector<std::string> files;
	DIR *dir;
	struct dirent *ptr;
  dir = opendir(path.c_str());
  HICE_CHECK_NOTNULL(dir);
 
	while ((ptr = readdir(dir)) != NULL) {
    ///current dir OR parrent dir
    const char* const fname = ptr->d_name;
		if (strcmp(fname, ".") == 0) continue;
		if (strcmp(fname, "..") == 0) continue;
    if (strlen(fname) < format.size()) continue;
    if (ptr->d_type != 8) continue;
    if (strcmp(fname + strlen(fname) - format.size(), format.c_str()) != 0) continue;
    std::string strFile;
    strFile = path;
    strFile += "/";
    strFile += ptr->d_name;
    files.push_back(strFile);
	}
	closedir(dir);
  return files;
}

}