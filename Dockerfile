FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

WORKDIR /workspace

RUN apt-get update 
RUN apt-get install -y software-properties-common wget curl git
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.10 python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt-get clean

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt-get -y purge python3.8
RUN apt-get -y autoremove

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENTRYPOINT ["/bin/bash"]
