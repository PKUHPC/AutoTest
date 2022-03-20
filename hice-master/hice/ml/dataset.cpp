#include "hice/ml/dataset.h"
#include <fstream>
#include <sstream>
#include "hice/core/tensor_printer.h"

namespace hice {

inline char *find_last_line(char *ptr, char *begin) {
  while (ptr != begin && *ptr != '\n') --ptr;
  return ptr;
}

inline int find_num_line_break(char *ptr, char *begin) {
  int num = 0;
  while (ptr != begin) {
    if (*ptr == '\n') num++;
    --ptr;
  };
  return num;
}

int dataset_num_dim(std::ifstream &ifs, int &num_of_data) {
  int buffer_size = 16 << 20;
  char *buffer = (char *)malloc(buffer_size);
  int num_line_break = 0;
  int max_features = 0;
  while (ifs) {
    char *head = buffer;
    ifs.read(buffer, buffer_size);
    size_t size = ifs.gcount();
    size_t sbegin = 0;
    size_t send = size - 1;
    char *pbegin = find_last_line(head + sbegin, head);
    char *pend = find_last_line(head + send, head);
    num_line_break += find_num_line_break(head + send, head + sbegin);
    ifs.seekg(pend - head - send, std::ios_base::cur);
    char *lbegin = pbegin;
    char *lend = lbegin;
    while (lend != pend) {
      lend = lbegin + 1;
      while (lend != pend && *lend != '\n') {
        ++lend;
      }
      char *last_word = lend;
      for (; last_word != lbegin;) {
        if ((*last_word) == ' ' && (lend - last_word) > 1)
          break;
        else
          last_word--;
      }
      std::string new_last_word(last_word, lend);
      std::stringstream last(new_last_word);
      std::string tuple;
      int i;
      float v;
      last >> tuple;
      sscanf(tuple.c_str(), "%d:%f", &i, &v);
      if (i > max_features) {
        max_features = i;
      }
      lbegin = lend;
    }
  }
  num_of_data = num_line_break;
  return max_features;
}

void load_data(std::ifstream &ifs, int num_of_data, int num_of_dim,
                 Tensor &value, Tensor &label) {
  int buffer_size = 16 << 20;
  char *buffer = (char *)malloc(buffer_size);
  auto value_data = value.mutable_data<float>();
  auto label_data = label.mutable_data<int32_t>();
  while (ifs) {
    int num_count = 0;
    char *head = buffer;
    ifs.read(buffer, buffer_size);
    size_t size = ifs.gcount();
    size_t sbegin = 0;
    size_t send = size - 1;
    char *pbegin = find_last_line(head + sbegin, head);
    char *pend = find_last_line(head + send, head);
    ifs.seekg(pend - head - send, std::ios_base::cur);
    char *lbegin = pbegin;
    char *lend = lbegin;
    int tlabel = 0;
    while (lend != pend) {
      int dim_count = 0;
      lend = lbegin + 1;
      while (lend != pend && *lend != '\n') {
        ++lend;
      }
      std::string line(lbegin, lend);
      std::stringstream ss(line);
      ss >> tlabel;
      label_data[num_count] = tlabel;
      std::string tuple;
      int prev_i = 0;
      while (ss >> tuple) {
        int i;
        float v;
        sscanf(tuple.c_str(), "%d:%f", &i, &v);
        for (int j = 1; j < i - prev_i; j++) {
          value_data[num_count * num_of_dim + dim_count] = 0;
          dim_count++;
        }
        value_data[num_count * num_of_dim + dim_count] = v;
        dim_count++;
        prev_i = i;
      }
      for (int j = 0; j < num_of_dim - prev_i; j++){
        value_data[num_count * num_of_dim + dim_count] = 0;
        dim_count++; 
      }
      lbegin = lend;
      num_count++;
    }
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> load_dataset(std::string dataset_name) {
  // Default training and testing dataset are both on CPU.
  // The deafault datatype for label is int.
  Tensor ref(device(kCPU).dtype(kFloat));
  Tensor query(device(kCPU).dtype(kFloat));
  Tensor ref_label(device(kCPU).dtype(kInt32));
  Tensor query_label(device(kCPU).dtype(kInt32));
  int num_of_ref, num_of_query, num_of_feature;

  // Check whether the dataset is supported.
  std::string train_file_name = "datasets/" + dataset_name;
  std::string test_file_name = "datasets/" + dataset_name + ".t";
  std::ifstream train_ifs(train_file_name, std::ifstream::binary);
  std::ifstream test_ifs(test_file_name, std::ifstream::binary);
  HICE_CHECK_INTERNAL(train_ifs.is_open() || test_ifs.is_open())
      << "Input dataset is not supported.";
  // Get the number of ref, query and feature.
  num_of_feature = std::max(dataset_num_dim(train_ifs, num_of_ref),
                       dataset_num_dim(test_ifs, num_of_query));
  train_ifs.clear();
  test_ifs.clear();
  train_ifs.seekg(0, std::ios::beg);
  test_ifs.seekg(0, std::ios::beg);
  ref.resize({num_of_ref, num_of_feature});
  query.resize({num_of_query, num_of_feature});
  ref_label.resize({num_of_ref, 1});
  query_label.resize({num_of_query, 1});
  load_data(train_ifs, num_of_ref, num_of_feature, ref, ref_label);
  load_data(test_ifs, num_of_query, num_of_feature, query, query_label);
  return std::make_tuple(ref, query, ref_label, query_label);
}

}  // namespce hice
