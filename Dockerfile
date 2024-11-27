FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
   apt-get install -y  --no-install-recommends  \
    ffmpeg \
    gcc \
    git \
    htop \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ncdu \
    openssh-client \
    openssh-server \
    psmisc \
    rsync \
    screen \
    tmux \
    unzip \
    vim \
    wget

RUN pip install onnx onnxscript torchinfo imageio imageio-ffmpeg==0.4.4 \
    click==8.1.7 \
    h5py==3.10.0 \
    matplotlib==3.8.2 \
    numpy==1.26.3 \
    pandas==2.2.0 \
    kaleido==0.2.1 \
    plotly==5.24.1 \
    plotly-express==0.4.1 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    seaborn==0.13.2 \
    torchvision==0.17.0

# setup ssh
RUN ssh-keygen -A
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

# Make the root user's home directory /myhome (the default for run.ai),
# and allow to login with password 'root'.
RUN echo 'root:root' | chpasswd
RUN sed -i 's|:root:/root:|:root:/myhome:|' /etc/passwd

# start ssh
ENTRYPOINT service ssh start && /bin/bash && /opt/conda/bin/conda init
