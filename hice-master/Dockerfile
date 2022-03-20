FROM nvidia/cuda:9.2-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Install CUDA and CUDNN
ENV CUDNN_VERSION 7.4.2.24
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install tools
RUN apt-get update && apt-get install -y \
    cpio \
    git \
    man \
    python \
    python-pip \
    vim \
    wget \
    openssh-server

# Enable ssh
RUN systemctl status ssh && \
    systemctl enable ssh && \
    systemctl start ssh && \
    ufw allow ssh && \
    ufw enable && \
    ufw status

# Install cmake
RUN apt remove --purge --auto-remove cmake && \
    cd /tmp && \
    wget https://cmake.org/files/v3.14/cmake-3.14.1-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.14.1-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \ 
    cmake --version

# Install MKL
RUN cd /tmp && \
  wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11306/l_mkl_2017.2.174.tgz && \
  tar -xzf l_mkl_2017.2.174.tgz && \
  cd l_mkl_2017.2.174 && \
  sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
  ./install.sh -s silent.cfg && \
  cd .. && \
  rm -rf *
RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
  ldconfig && \
  echo ". /opt/intel/bin/compilervars.sh intel64" >> /etc/bash.bashrc