FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG APPLICATION_DIRECTORY=/home/challenger/sfi_convtasnet

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV PYTORCH_VERSION=1.5.0
# ENV TORCHVISION_VERSION=0.6.0
# ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
# ENV NCCL_VERSION=2.4.8-1+cuda10.1
# COPY install.txt /root/src
# COPY torchhub.txt /root/src

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ENV PYTHON_VERSION=3.8

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
    	ffmpeg \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

RUN apt-get install -y python${PYTHON_VERSION}-distutils && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
    
    # && \
    # pip install poetry

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}
