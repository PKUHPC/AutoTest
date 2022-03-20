# HICE
High-performance Intelligent Computation Engine(HICE)

<!-- TOC -->

- [HICE](#hice)
  - [Environment Requirement](#environment-requirement)
  - [Installation](#installation)
    - [Clone source](#clone-source)
    - [Build and Install](#build-and-install)
    - [Test](#test)
  - [Other](#other)
    - [Importing HICE into your project](#importing-hice-into-your-project)
    - [Getting Started](#getting-started)

<!-- /TOC -->


## Environment Requirement

| Dependency | Version required    |
| ---------- | ------------------- |
| gcc        | `5.0` or higher     |
| mkl        | `2018.01` or higher |
| cuda       | `11.0` or higher     |
| cudnn      | `7.6` or higher     |
| CMake      | `3.11` or higher    |
| TVM        | `0.8` or higher    |

## Installation

### Clone source
Assume that HICE is going to be cloned into `${HICE_SOURCE_DIR}` and be installed into `${HICE_INSTALL_DIR}`.
```bash
git clone https://github.com/pku-hpc/hice.git
git submodule update --init --recursive
```

### Build and Install
```bash
cd ${HICE_SOURCE_DIR}
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${HICE_INSTALL_DIR}
make -j ${nproc}
make install -j ${nproc}
```

### Test
Unit test:
```bash
./bin/main
```
Sparse performance test:
```bash
./bin/spmv_test ${dataset_name}
./bin/spmm_test ${dataset_name}
./bin/spgemm_test ${dataset_name} ${n_cols}
```
`NOTE:` The datasets could be downloaded from [http://yifanhu.net/GALLERY/GRAPHS/search.html](http://yifanhu.net/GALLERY/GRAPHS/search.html)


## Other

### Importing HICE into your project
Clone HICE and add following codes in your CMakeLists.txt.
```cmake
# add HICE
add_subdirectory(${HICE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/hice)
target_link_libraries(your_target hice::hice)
```

### Integrated into Dragon
Install HICE into ${PATH_PREFIX}/hice/install_dragon_naive

Install TVM8.0 into ${path_prefix}/tvm, and export environment variable: TVM_HOME=${path_prefix}/tvm

Clone dragon(hice version) into ${PATH_PREFIX}/dragon, link: git@github.com:pku-hpc/dragon-for-hice.git

Compile and install dragon

### Getting Started
Here is an example, you can find it in `${HICE_SOURCE_DIR}/examples/matmul.cpp`:
```c++
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/matmul.h"

using namespace hice;

int main() {
  TensorPrinter tp;
  
  // CPU matmul 
  std::cout << "==============================" << std::endl;
  std::cout << "     CPU matmul example       " << std::endl;
  std::cout << "==============================" << std::endl;
  Tensor h_mat1 = full({4, 4}, 1, device(kCPU).dtype(kDouble));
  Tensor h_mat2 = full({4, 4}, 1, device(kCPU).dtype(kDouble));
  Tensor h_mat3 = matmul(h_mat1, h_mat2);
  tp.print(h_mat3);

  // CUDA matmul 
  std::cout << "==============================" << std::endl;
  std::cout << "     CUDA matmul example      " << std::endl;
  std::cout << "==============================" << std::endl;
  Tensor d_mat1 = full({4, 4}, 1, device(kCUDA).dtype(kDouble));
  Tensor d_mat2 = full({4, 4}, 1, device(kCUDA).dtype(kDouble));
  Tensor d_mat3 = matmul(d_mat1,d_mat2);
  tp.print(d_mat3);
}

```