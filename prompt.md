# Optimized Triton GEMM Kernel for Small Batch Transformer Inference on Low-Resource Hardware

**Robert Zhang**  
zhan4808@purdue.edu

## Introduction

Transformers have revolutionized natural language processing and computer vision with their ability to learn complex patterns, but their computational demands, particularly for general matrix multiplication (GEMM), create challenges for real-time inference in time-sensitive applications like fraud detection and on hardware with limited resources. While much effort has gone into optimizing inference latency for datacenter platforms, enabling transformer inference in low-resource environments is equally critical, as deploying advanced AI models on edge devices and embedded systems requires real-time performance under strict power and memory constraints.

This project aims to design an optimized GEMM kernel for small-batch transformer inference using Triton, a domain-specific GPU programming language developed by OpenAI to create highly efficient kernels for large language model architectures. The approach focuses on reducing latency, improving energy efficiency, and adapting to hardware-specific constraints, with potential applicability to non-GPU architectures.

## Optimizations

This project uses the following techniques to optimize GEMM operations tailored to the self-attention and feedforward mechanisms of transformers:

1. Memory transfer overhead is minimized by leveraging hierarchical memory structures such as shared memory and L1/L2 caches on GPUs or scratchpad memory on non-GPU architectures. This reduces redundant data transfers of matrices Q, K^T, V used in self-attention, and weights W1, W2 used in feedforward, accelerating Q · K^T and V matrix multiplications and W₁ · X + b₁ operations.

2. Reduced precision arithmetic with INT4 is implemented to achieve low-latency computation with minimal accuracy loss. This is particularly effective for scaled dot product computations Q × K^T / √d_k in self-attention and reduces energy cost in weight-heavy feedforward layers.

3. Tiling and blocking divide matrices into submatrices that fit within local memory, improving data reuse and minimizing off-chip memory access. For transformer workloads, this ensures efficient handling of Q, K^T, V matrices in self-attention and large weight matrices W₁, W₂ in feedforward layers, particularly for common transformer dimensions such as d_hidden = 2048.

4. A weight-stationary dataflow retains weight matrices in local memory during computation, reducing power consumption and data movement. In self-attention, K^T and V are reused across all queries Q, so stationary weights allows efficient multi-head attention. For feedforward layers, W₁ and W₂ weights can be stored in shared memory for repeated use, drastically reducing overhead of fetching large matrices from global memory.

Possible alternate or further optimizations include kernel fusion of GEMM and bias addition or activations as well as tensor layout optimization for memory access patterns.

## Impact

These optimizations make real-time transformer inference more feasible and practical in edge devices and embedded systems, bridging the gap between theoretical advancements and practical deployment. The end result is an __open-source, small-batch transformer inference-optimized GEMM kernel that is transformer model-optimized, scalable, energy-efficient, and hardware-adaptable__.

The work promises outcomes such as novel optimization techniques, publishable research, and practical tools for hardware-software co-design. Additionally, I hope that this project can also set a more accessible platform for future interest and growth in GPU programming, Triton, and AI system optimization.


Choosing Small-Batch Optimization
Focusing on optimizing GEMM latency for small batches, especially considering the limited memory on many edge devices, is a very relevant direction given the current trend of deploying large models on resource-constrained hardware. This aligns well with the growing demand for efficient inference in real-world applications.

Memory Optimization Considerations
Techniques like leveraging shared memory, double-buffering, and pipelined execution are well-established optimization strategies. However, if you're using Triton, there’s an important caveat:

Due to the characteristics of Triton's programming model, the current version of the language does not provide explicit control over shared memory.
Instead, you can achieve reasonable caching performance by using L2 cache efficiently, which often involves swizzle mechanisms for data layout optimization.
For more fine-grained memory control, it might require modifications to Triton’s compiler passes, which can be complex. (That said, I haven’t reviewed the latest updates to Triton compile recently, so there might be new features that support this better.)
On Low-Precision Arithmetic
Regarding your work on reduced precision, I’m curious if your goal is to:

Simply implement a known low-precision kernel (where the algorithm is already established and backed by papers).
Or, to develop a novel quantization algorithm.
If it’s the former, one idea could be to identify a strong quantization algorithm (with no Triton implementation yet but proven effective in papers) and implement it. For the latter, creating a new algorithm would undoubtedly be much more challenging.
Small-Batch Specific Algorithms
Since your focus is on small-batch inference, aside from quantization, there are other GEMM-related optimizations like split-K or stream-K algorithms, which have theoretical advantages in small-batch scenarios. However, as noted in point 2, some Triton language limitations may also apply here.

Further Resources
Lastly, I’d recommend checking out a few relevant blogs and implementations that could serve as inspiration or reference:

https://pytorch.org/blog/accelerating-gemms-triton/
https://pytorch.org/blog/accelerating-triton/
https://pytorch.org/blog/accelerating-llama3/
https://pytorch.org/blog/accelerating-generative-ai-2/