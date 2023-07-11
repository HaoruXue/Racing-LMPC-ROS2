FROM osrf/ros:galactic-desktop
LABEL maintainer="Haoru Xue <haorux@andrew.cmu.edu>"

SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade -y && \
    apt install -y \
    ca-certificates \
    build-essential \
    git \
    iputils-ping \
    nano \
    gdb \
    vim \
    curl \
    wget \
    rsync \
    openssh-client \
    python3-pip \
    python3-venv \
    python-pip

RUN python3 -m pip install casadi

# casadi
RUN apt install -y gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends && \
    apt install -y --no-install-recommends coinor-libipopt-dev libslicot-dev && \
    cd /opt && git clone https://github.com/casadi/casadi.git -b main casadi && \
    cd casadi && mkdir build && cd build && \
    cmake -DWITH_IPOPT=ON -DWITH_SLICOT=ON -DWITH_LAPACK=ON .. && make && \
    make install && \
    echo export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" >> /etc/bash.bashrc

# clone private repos
RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh \
    git clone git@github.com:MPC-Berkeley/barc_core.git && \
    cd barc_core/ws/src && \
    vcs import < ../../repos/devel-ed.repos && \
    vcs import < ../../repos/deps-foxglove.repos