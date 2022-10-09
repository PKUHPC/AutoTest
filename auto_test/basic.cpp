#include "auto_test/basic.h"
#include <random>

namespace aitisa_api {

const DataType aitisa_dtypes[10] = {kInt8,   kUint8, kInt16,  kUint16, kInt32,
                                    kUint32, kInt64, kUint64, kFloat,  kDouble};
const Device aitisa_devices[2] = {{DEVICE_CPU, 0}, {DEVICE_CUDA, 0}};

template <typename DATATYPE>
inline void natural_assign_int(DATATYPE* data, unsigned int nelem) {
  for (unsigned int i = 0; i < nelem; i++) {
    data[i] = (DATATYPE)i + 1;
  }
}
template <typename DATATYPE>
inline void natural_assign_float(DATATYPE* data, unsigned int nelem) {
  for (unsigned int i = 0; i < nelem; i++) {
    data[i] = (DATATYPE)i * 0.1 + 0.1;
  }
}
void natural_assign(void* data, unsigned int len, int dtype) {
  switch (dtype) {
    case 0:
      natural_assign_int((int8_t*)data, len / sizeof(int8_t));
      break;
    case 1:
      natural_assign_int((uint8_t*)data, len / sizeof(uint8_t));
      break;
    case 2:
      natural_assign_int((int16_t*)data, len / sizeof(int16_t));
      break;
    case 3:
      natural_assign_int((uint16_t*)data, len / sizeof(uint16_t));
      break;
    case 4:
      natural_assign_int((int32_t*)data, len / sizeof(int32_t));
      break;
    case 5:
      natural_assign_int((uint32_t*)data, len / sizeof(uint32_t));
      break;
    case 6:
      natural_assign_int((int64_t*)data, len / sizeof(int64_t));
      break;
    case 7:
      natural_assign_int((uint64_t*)data, len / sizeof(uint64_t));
      break;
    case 8:
      natural_assign_float((float*)data, len / sizeof(float));
      break;
    case 9:
      natural_assign_float((double*)data, len / sizeof(double));
      break;
    default:
      break;
  }
}

template <typename DATATYPE>
inline void random_assign_int(DATATYPE* data, unsigned int nelem) {
  std::default_random_engine gen(/*seed*/ 0);
  std::uniform_int_distribution<DATATYPE> dis(0, 10);
  for (unsigned int i = 0; i < nelem; i++) {
    data[i] = dis(gen);
  }
}
template <typename DATATYPE>
inline void random_assign_float(DATATYPE* data, unsigned int nelem) {
  std::default_random_engine gen(/*seed*/ 0);
  std::normal_distribution<DATATYPE> dis(0, 1);
  for (unsigned int i = 0; i < nelem; i++) {
    data[i] = dis(gen);
  }
}
void random_assign(void* data, unsigned int len, int dtype) {
  switch (dtype) {
    case 0:
      random_assign_int((int8_t*)data, len / sizeof(int8_t));
      break;
    case 1:
      random_assign_int((uint8_t*)data, len / sizeof(uint8_t));
      break;
    case 2:
      random_assign_int((int16_t*)data, len / sizeof(int16_t));
      break;
    case 3:
      random_assign_int((uint16_t*)data, len / sizeof(uint16_t));
      break;
    case 4:
      random_assign_int((int32_t*)data, len / sizeof(int32_t));
      break;
    case 5:
      random_assign_int((uint32_t*)data, len / sizeof(uint32_t));
      break;
    case 6:
      random_assign_int((int64_t*)data, len / sizeof(int64_t));
      break;
    case 7:
      random_assign_int((uint64_t*)data, len / sizeof(uint64_t));
      break;
    case 8:
      random_assign_float((float*)data, len / sizeof(float));
      break;
    case 9:
      random_assign_float((double*)data, len / sizeof(double));
      break;
    default:
      break;
  }
}

void full_float(Tensor t, const float value) {
  int64_t size = aitisa_tensor_size(t);
  auto* data = (float*)aitisa_tensor_data(t);
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
}

#ifdef AITISA_API_GENERATE_FIGURE
void draw_fig_fun(const time_map& m, const std::string& filename) {
  if (Py_IsInitialized()) {
    PyObject* p_module = nullptr;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../../script/')");
    p_module = PyImport_ImportModule("draw_fig");
    if (p_module) {
      PyObject* p_func = PyObject_GetAttrString(p_module, "draw_fig");
      PyObject* p_args = PyTuple_New(4);

      PyObject* test_case_list = PyList_New(0);
      PyObject* time_list = PyList_New(0);
      PyObject* op_kind_list = PyList_New(0);

      for (const auto& kv : m) {
        PyList_Append(test_case_list, Py_BuildValue("s", kv.first.c_str()));
        PyList_Append(time_list, Py_BuildValue("d", std::get<0>(kv.second)));
        PyList_Append(op_kind_list, Py_BuildValue("s", "aitisa"));

        PyList_Append(test_case_list, Py_BuildValue("s", kv.first.c_str()));
        PyList_Append(time_list, Py_BuildValue("d", std::get<1>(kv.second)));
        PyList_Append(op_kind_list, Py_BuildValue("s", "user"));
#ifdef AITISA_API_PYTORCH
        PyList_Append(test_case_list, Py_BuildValue("s", kv.first.c_str()));
        PyList_Append(time_list, Py_BuildValue("d", std::get<2>(kv.second)));
        PyList_Append(op_kind_list, Py_BuildValue("s", "torch"));
#endif
      }

      PyTuple_SetItem(p_args, 0, test_case_list);
      PyTuple_SetItem(p_args, 1, time_list);
      PyTuple_SetItem(p_args, 2, op_kind_list);
      PyTuple_SetItem(p_args, 3, Py_BuildValue("s", filename.c_str()));

      PyEval_CallObject(p_func, p_args);

    } else {
      printf("python import failed...\n");
    }
  } else {
    printf("python initialized failed...\n");
  }
}
#endif

}  // namespace aitisa_api