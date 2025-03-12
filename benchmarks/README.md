# Benchmarks

This directory includes a series of CUDA source files that incrementally improve benchmarking methodology for GPU kernels.

> [!IMPORTANT]  
> This is the code for the blog post ["How to Benchmark CUDA"](https://guillesanbri.com/CUDA-Benchmark/) which provides explanations and results for each methodology change. Worth taking a look!

## Final Template

The most complete template, along with some guidelines, can be found in `benchmark_00_template.cu`.

## Files Overview

### **benchmark_01_wrong.cu**

Starting point, this is not really a valid way to time CUDA kernels at all.

**WIP**