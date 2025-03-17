FROM ubuntu:20.04

<<<<<<< HEAD
# Set environment variables to ensure non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
=======
WORKDIR /workspace
>>>>>>> 6f269eabf (17/03/2025; json; add Dockerfile)

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

<<<<<<< HEAD
# Install Python 3.10 dependencies
RUN pip install numpy scipy matplotlib

# Set working directory for project
WORKDIR /root

# Clone the GitHub repository
RUN git clone -b main https://github.com/BolunZhangzbl/NS3-DRL-ORAN.git json

# Set the working directory to the cloned repo
WORKDIR /root/json

# Optionally: If there are any build steps needed, add them here, e.g.:
# RUN mkdir build && cd build && cmake ..
# RUN make

# Expose necessary ports (if required for the app)
# EXPOSE 8080  # Uncomment and modify according to your application
=======
# Set working directory for project
WORKDIR /workspace

# Clone the GitHub repository
RUN git clone -b main https://github.com/BolunZhangzbl/NS3-DRL-ORAN.git ns-3-dev

# Set the working directory to the cloned repo
WORKDIR /workspace/ns-3-dev

RUN ./waf configure --enable-tests --enable-examples
RUN ./waf build

WORKDIR /workspace
>>>>>>> 6f269eabf (17/03/2025; json; add Dockerfile)

# Set the default command to run when the container starts (e.g., bash or other commands)
CMD ["/bin/bash"]
