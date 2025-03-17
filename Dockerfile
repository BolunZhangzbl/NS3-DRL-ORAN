FROM ubuntu:20.04

WORKDIR /workspace

# Update and install essential packages
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

WORKDIR /workspace

RUN git clone -b json https://github.com/BolunZhangzbl/NS3-DRL-ORAN.git ns-3-dev

# Set the working directory to the cloned repo
WORKDIR /workspace/ns-3-dev

RUN ./waf configure --enable-tests --enable-examples
RUN ./waf build

RUN pip install -e .

RUN chmod +x run_both.sh

# Set the default command to run when the container starts (e.g., bash or other commands)
CMD ["/bin/bash"]