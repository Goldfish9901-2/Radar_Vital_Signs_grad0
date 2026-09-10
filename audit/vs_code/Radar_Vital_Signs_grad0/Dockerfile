FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# ENV CONDA_DIR=/opt/conda
# ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN cat > /etc/apt/sources.list << EOF
deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
EOF

RUN apt-get update && apt-get install -y \
    curl \
    git \
    bzip2 \
    ca-certificates \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
SHELL ["/bin/bash", "-lc"]

RUN /opt/venv/bin/pip install --upgrade\
    -i https://mirrors.aliyun.com/pypi/simple \
    pip uv

RUN /opt/venv/bin/uv pip install \
    --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple \
     numpy pandas neurokit2 matplotlib scipy  tqdm torch torchvision torchaudio 

WORKDIR /Radar_Vital_Signs

CMD python -c "import torch, numpy, scipy, matplotlib, neurokit2; \
print('torch=', torch.__version__); \
print('cuda available=', torch.cuda.is_available()); \
print('device count=', torch.cuda.device_count()); \
print('numpy=', numpy.__version__); \
print('scipy=', scipy.__version__); \
print('matplotlib=', matplotlib.__version__); \
print('neurokit2=', neurokit2.__version__)"
