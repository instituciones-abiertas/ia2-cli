FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

ARG PIP_VERSION
ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG TORCH_VERSION

# OS packages
RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
    && apt-get update --fix-missing; exit 0

RUN apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y --no-install-recommends \
    curl \
    wget \
    git-core \
    htop \
    unzip \
    zsh \
    vim \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    libtool \
    libleptonica-dev \
    libltdl-dev \
    pkg-config \
    libgl1-mesa-glx \
    python3-setuptools \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

# Set default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /tmp

# Install pip
RUN wget https://github.com/pypa/pip/archive/refs/tags/${PIP_VERSION}.zip \
    && unzip ${PIP_VERSION}.zip \
    && cd pip-${PIP_VERSION} \
    && python setup.py install

# Install pytorch
RUN --mount=type=cache,target=/root/.cache \
    pip install torch==${TORCH_VERSION}+${CUDA_VERSION} \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

#Install Spacy
RUN pip install -U pip setuptools wheel \
    pip install -U spacy[${CUDA_VERSION}] \
    pip install spacy-transformers \
    pip install cupy-wheel

# install requirements
COPY ./src/ia2 /tmp/ia2
RUN --mount=type=cache,target=/root/.cache \
    pip install ./ia2

RUN rm -rf /tmp/*

WORKDIR /root