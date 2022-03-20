#pragma once

#include <cstdarg>
#include <memory>
#include <vector>
#include <Python.h>

namespace hice {

typedef PyObject* HicePyObject;

/// NOTE: Call Py_Finalize or Py_CLEAR(module) more than once might cause 
//  crash because of numpy's bug. PythonEnvManager call Py_Initialize only 
//  once and keep some imported python modules, which modules will be freed 
//  when exit. See more at 
//  https://stackoverflow.com/questions/7676314/py-initialize-py-finalize-not-working-twice-with-numpy/7676916#7676916
class PythonEnvManager {
typedef std::shared_ptr<PythonEnvManager> PythonEnvManagerPtr;

public:
  static PythonEnvManagerPtr getInstance() {
    static PythonEnvManagerPtr instance(new PythonEnvManager());
    return instance;
  }

  bool initialize() {
    if (!inited_) {
      Py_Initialize();
      inited_ = Py_IsInitialized();
      state_ = PyGILState_Ensure();
      PyRun_SimpleString("import sys");
      PyRun_SimpleString("sys.path.append('/home/amax101/hice/likesen/hice/hice/tvm/py_modules')");
    } else {
      state_ = PyGILState_Ensure();
    }
    return inited_;
  }

  bool finalize() { 
    for (HicePyObject& ptr : py_ptrs_) {
      if (ptr) Py_CLEAR(ptr);
    }
    py_ptrs_.clear();

    PyGILState_Release(state_);
    return true;
  }

  bool trace(HicePyObject ptr) { py_ptrs_.push_back(ptr); }

  bool inited() { return inited_; }

  ~PythonEnvManager() {
    if (inited_) {
      Py_Finalize();
    }
  }

private:
  PythonEnvManager(): inited_(false), state_(PyGILState_LOCKED), py_ptrs_() { }
  PythonEnvManager(PythonEnvManager const&) = delete;
  PythonEnvManager& operator=(PythonEnvManager const&) = delete;

  bool inited_;
  /// NOTE: hice_tvm module is supposed to be called by higher level
  //  framework, which might be runing under multi-thread python envs,
  //  so GIL is required to call tvm-module in python. Or segfault occurs.
  PyGILState_STATE state_;
  std::vector<HicePyObject> py_ptrs_;
};

inline void HICE_Py_Initialize() {
  HICE_CHECK(PythonEnvManager::getInstance()->initialize());
}

inline bool HICE_Py_IsInitialized() { 
  return PythonEnvManager::getInstance()->inited();
}

inline void HICE_Py_Finalize() { 
  HICE_CHECK(PythonEnvManager::getInstance()->finalize());
}

inline void HICE_PyDict_SetItemString(HicePyObject p, const char *key, HicePyObject val) {
  HICE_CHECK_EQ(0, PyDict_SetItemString(p, key, val));
}

inline void HICE_PyTuple_SetItem(HicePyObject p, Py_ssize_t pos, HicePyObject o) {
  // HICE traces every py_obj from HICE_Py_APIs. So Py_XINCREF is need here to avoid 
  // HICE clear HicePyObject o twice, see more at
  // https://docs.python.org/3.6/c-api/intro.html#reference-count-details
  Py_XINCREF(o);
  HICE_CHECK_EQ(0, PyTuple_SetItem(p, pos, o));
}

inline void HICE_PyRun_SimpleString(const char *command) {
  HICE_CHECK_EQ(0, PyRun_SimpleString(command));
}

inline HicePyObject HICE_PyObject_GetAttrString(HicePyObject o, const char* attr_name) {
  HicePyObject ptr = PyObject_GetAttrString(o, attr_name);
  PythonEnvManager::getInstance()->trace(ptr);
  return ptr;
}

inline HicePyObject HICE_PyImport_ImportModule(const char* name) {
  HicePyObject ptr = PyImport_ImportModule(name);
  PythonEnvManager::getInstance()->trace(ptr);
  return ptr;
}

inline HicePyObject HICE_PyDict_New() {
  HicePyObject ptr = PyDict_New();
  PythonEnvManager::getInstance()->trace(ptr);
  return ptr;
}

inline HicePyObject HICE_PyTuple_New(Py_ssize_t len) {
  HicePyObject ptr = PyTuple_New(len);
  PythonEnvManager::getInstance()->trace(ptr);
  return ptr;
}

inline HicePyObject HICE_Py_BuildValue(const char *format, ...) {
  va_list vargs;
  va_start(vargs, format);
  HicePyObject ptr = Py_VaBuildValue(format, vargs);
  PythonEnvManager::getInstance()->trace(ptr);
  va_end(vargs);
  return ptr;
}

inline HicePyObject HICE_PyObject_CallObject(HicePyObject callable_object, HicePyObject args) {
  HicePyObject ptr = PyObject_CallObject(callable_object, args);
  PythonEnvManager::getInstance()->trace(ptr);
  return ptr;
}

} // namespace hice