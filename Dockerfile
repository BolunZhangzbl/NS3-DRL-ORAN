ARG CUDA_VERSION=12.1.1

# --------------------------
# Stage 1: Build NS-3
# --------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git nano nlohmann-json3-dev curl sudo \
    && apt-get install -y build-essential cmake libsctp-dev autoconf automake libtool bison flex libboost-all-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Copy source and build NS-3
WORKDIR /ns-3-dev
COPY . /ns-3-dev

RUN ./waf configure --enable-tests --enable-examples \
    && ./waf build

# --------------------------
# Stage 2: Runtime Image
# --------------------------
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy buil3 ns-3 files from builder
COPY --from=builder /ns-3-dev /ns-3-dev
WORKDIR /ns-3-dev

# Install NS-3 Python bindings
RUN pip install -e .

# Make the run script executable
RUN chmod +x run_both.sh

# Set the default command
CMD ["/bin/bash"]

# ENTRYPOINT ["/ns-3-dev/run_both.sh"]