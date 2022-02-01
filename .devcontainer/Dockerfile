# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile


# Development container
ARG VARIANT="ubuntu-20.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT} AS dev

# Download NLopt
ADD https://github.com/stevengj/nlopt/archive/refs/tags/v2.7.1.tar.gz /root/downloads/

# Install packages and their dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Install Meson
    && apt-get -y install --no-install-recommends \
    python3 python3-pip python3-setuptools python3-wheel ninja-build \
    && pip3 install meson \
    # Install NLopt
    && cd /root/downloads && tar -xvf v2.7.1.tar.gz && cd nlopt-2.7.1/ \
    && cmake . && make && sudo make install \
    # Boost
    && apt-get -y install --no-install-recommends libboost-all-dev \
    # Pkg-config
    pkg-config

# Control automatic EOL conversion so that git inside container does not report all files changed
RUN git config --global core.autocrlf true


# Build container
FROM dev as build

WORKDIR /root/examplecp/
COPY . .

# Build according to meson.build
RUN meson setup builddir && cd builddir && ninja


# Release container
FROM ubuntu:20.04 AS release

WORKDIR /root/examplecp/
COPY --from=build /root/examplecp/builddir/main ./
COPY --from=build /lib/x86_64-linux-gnu/libgomp.so.1 /lib/x86_64-linux-gnu/libgomp.so.1

# [TODO] The main itself does not know how to make dir if one does not exist, please 
# [TODO] adjust the source code in future.
RUN mkdir /root/output

CMD ["./main"]