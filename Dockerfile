ARG CUDA_VERSION=12.1.1

# Stage 1: Build stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install nlohmann/json library (required for ns3 example)
RUN apt-get update && apt-get install -y nlohmann-json3-dev

# Set the working directory
WORKDIR /workspace

# Install essential packages for building NS3
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libsctp-dev \
    autoconf \
    automake \
    libtool \
    bison \
    flex \
    libboost-all-dev \
    python3-pip \
    g++-9 \
    && apt-get clean

# Install pip for Python 3.10
RUN python3 -m pip install --upgrade pip

# Copy the local repository into the build stage
COPY . /workspace/ns-3-dev

# Build NS3
WORKDIR /workspace/ns-3-dev
RUN ./waf configure --enable-tests --enable-examples
RUN ./waf build

# Stage 2: Final image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Copy only the necessary files from the builder stage
COPY --from=builder /workspace/ns-3-dev /workspace/ns-3-dev

# Install PyTorch with GPU support (if not already available in the base image)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio

# Install NS3 Python bindings
WORKDIR /workspace/ns-3-dev
RUN pip install -e .

# Make the run script executable
RUN chmod +x run_both.sh

# Set the default command
CMD ["/bin/bash"]