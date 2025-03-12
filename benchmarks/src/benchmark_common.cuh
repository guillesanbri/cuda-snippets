#ifndef BENCHMARK_COMMON_CUH
#define BENCHMARK_COMMON_CUH

#include <iostream>
#include <vector>
#include <random>

namespace benchmark {

    // Error Handling
    // TODO

    // Kernel 
    // (naive approach, no shared memory tiling or extra optimizations)
    __global__ 
    void matMulKernel(float *A, float *B, float *C, int m, int k, int n){
        // A * B = C
        // (m x k)(k x n) = (m x n)
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (row < m && col < n){
            float cValue = 0;
            for (int k_i=0; k_i < k; k_i++){
                cValue += A[row * k + k_i] * B[k_i * n + col];
            }
            C[row * n + col] = cValue;
        }

    }

    // Utility Functions
    inline void randomizeVector(std::vector<float>& v, float min=0, float max=100){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& v_i : v){
            v_i = dist(gen);
        }
    }

}

#endif // BENCHMARK_COMMON_CUH