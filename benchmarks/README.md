# Benchmarks

This directory includes a series of CUDA source files that incrementally improve benchmarking methodology for GPU kernels.

> [!IMPORTANT]  
> This is the code for the blog post ["How to Benchmark CUDA Kernels"](https://guillesanbri.com/CUDA-Benchmarks/) which provides explanations and results for each methodology change. Worth taking a look!

## Files Overview

### **benchmark_01_wrong.cu**

Starting point, this is not really a valid way to time CUDA kernels at all.

### **benchmark_02_sync.cu**

Adds GPU synchronization before measuring end of execution.

### **benchmark_03_cudaEvents.cu**

Replaces `chrono` timers with `cudaEvents`.

### **benchmark_04_coldStart.cu**

Launches the kernel a few times before measuring to remove cold start.

### **benchmark_05_coldStartDifferent.cu**

Randomizes the input data used in the warming up kernel launches and the benchmarked launch.

### **benchmark_06_coldStartEmptyKernel.cu**

Tests the effect of warming up with an empty kernel instead of using the kernel being benchmarked.

### **benchmark_07_statistics.cu**

Adds statistics over multiple runs to make results more stable.

### **benchmark_08_cacheDifferentData.cu**

Randomizes the input after every launch of the kernel being benchmarked.

### **benchmark_09_cacheFlush.cu**

Flushes L2 cache memory between benchmarked runs.