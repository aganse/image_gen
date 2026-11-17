# Same base image as flow_models, for parity of the job-submission machinery.
# PyTorch's pip wheels bundle their own CUDA/cuDNN runtime libraries, so the
# only hard requirement is that the wheel's CUDA build (cu126 below) is
# supported by the host's NVIDIA driver; the CUDA libs in this base image are
# not actually used by torch.  If you change the base image's CUDA version,
# keep TORCH_CUDA in step (see https://pytorch.org/get-started/locally/).
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=12.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# DEVICE=gpu|cpu is passed by `make local-build` and by CodeBuild (buildspec.yml).
ARG DEVICE=gpu
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCH_CUDA=cu126

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Install torch first from the PyTorch wheel index matching DEVICE, so that
# requirements.txt below finds it already satisfied (cpu wheels are far smaller).
RUN if [ "$DEVICE" = "cpu" ]; then TORCH_INDEX=cpu; else TORCH_INDEX=$TORCH_CUDA; fi \
    && pip3 install --no-cache-dir \
       torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
       --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

RUN pip3 install --no-cache-dir -r requirements.txt

# "Bake" the Inception-v3 weights used by the FID/KID/IS metrics into the
# image.  Training jobs run in a private subnet with no internet access, so
# they cannot download these at run time; CodeBuild / local builds can.
# (Any pretrained Hugging Face weights you want to use would be baked in the
# same way: download here into HF_HOME, then run with HF_HUB_OFFLINE=1.)
ENV TORCH_HOME=/opt/torch_home
RUN python3 -c "from torchmetrics.image.fid import FrechetInceptionDistance; FrechetInceptionDistance(feature=2048)"

COPY *.py /app/

COPY aws/job-support-common/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
