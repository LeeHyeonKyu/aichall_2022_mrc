ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="8.0"

RUN pip install matplotlib pillow tensorboardX tqdm wandb==0.12.14 icecream
RUN pip install scikit-learn==1.0
RUN pip install seaborn==0.11.2
RUN pip install timm==0.5.4
RUN pip install yacs

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim ffmpeg

WORKDIR /aichall_2022_mrc
RUN pip3 install transformers
RUN git config --global --add safe.directory /aichall_2022_mrc