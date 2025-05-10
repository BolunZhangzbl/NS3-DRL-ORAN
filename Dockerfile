ARG CUDA_VERSION=12.1.1

# Stage 1: Build stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder

WORKDIR /workspace

# Install essential packages for building NS3
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    cmake \
    libsctp-dev \
    autoconf \
    automake \
    libtool \
    bison \
    flex \
    libboost-all-dev \
    python3.10 \
    python3-pip \
    g++-9 \
    && apt-get clean

# Set python3.10 as the default python3 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3

# Install pip for Python 3.10
RUN python3 -m pip install --upgrade pip

# Copy the local repository into the build stage
COPY . /workspace/ns-3-dev

# Build NS3
WORKDIR /workspace/ns-3-dev
RUN ./waf configure --enable-tests --enable-examples
RUN ./waf build

# Stage 2: Final image
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

WORKDIR /workspace

# Copy only the necessary files from the builder stage
COPY --from=builder /workspace/ns-3-dev /workspace/ns-3-dev

# Install Python bindings
WORKDIR /workspace/ns-3-dev
RUN pip install -e .

# Install PyTorch with GPU support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Make the run script executable
RUN chmod +x run_both.sh

# Set the default command
CMD ["/bin/bash"]
