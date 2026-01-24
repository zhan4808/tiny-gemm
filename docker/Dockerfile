FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tiny-gemm

COPY requirements.txt /workspace/tiny-gemm/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch \
    && python3 -m pip install -r requirements.txt

COPY . /workspace/tiny-gemm

CMD ["bash"]
