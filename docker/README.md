# GPU Container Usage

This container is meant for NVIDIA GPUs with CUDA drivers installed on the host.

## Build

```
docker build -t tiny-gemm:cuda -f docker/Dockerfile .
```

## Run

```
docker run --gpus all -it --rm -v "$PWD":/workspace/tiny-gemm tiny-gemm:cuda
```

## Quick checks

```
python3 benchmark_fused_transformer.py --mode=seq_length --seq_lengths 128 256 512
python3 benchmark_gemm.py --m 512 --n 512 --k 512
```
