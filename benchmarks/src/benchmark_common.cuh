#ifndef BENCHMARK_COMMON_CUH
#define BENCHMARK_COMMON_CUH

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

#define CHECK_CUDA_ERROR(val) benchmark::check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() benchmark::checkLast(__FILE__, __LINE__)

namespace benchmark {

    // Cold start iterations
    constexpr int CS_ITERS = 1000;

    // Benchmark iterations
    constexpr int ITERS = 5000;

    // Error Handling
    // From https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
    void check(cudaError_t err, const char* const func, const char* const file,
            const int line)
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void checkLast(const char* const file, const int line)
    {
        cudaError_t const err{cudaGetLastError()};
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

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

    // Empty kernel
    __global__ void empty() {}

    // Utility Functions
    inline void randomizeVector(std::vector<float>& v, float min=0, float max=100){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& v_i : v){
            v_i = dist(gen);
        }
    }

    inline float percentile(std::vector<float>& v, float p){
        size_t n = v.size();
        std::sort(v.begin(), v.end());
        float f_index = (p / 100.0) * (n - 1);
        size_t lower = std::floor(f_index);
        size_t upper = std::ceil(f_index);
        if (lower == upper) return v[lower];
        return v[lower] + (v[upper] - v[lower]) * (f_index - lower);
    }

    inline float mean(std::vector<float>& v){
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    inline float standard_deviation(std::vector<float>& v){
        float _mean = mean(v);
        float sum_sq_differences = 0.0;
        for (auto i : v) sum_sq_differences += (i - _mean) * (i - _mean);
        return std::sqrt(sum_sq_differences / v.size());
    }

    inline void printResults(std::vector<float>& runtimes){
        std::sort(runtimes.begin(), runtimes.end());
        float _mean = mean(runtimes);
        float _std = standard_deviation(runtimes);
        float _median = percentile(runtimes, 50.0);
        float _min = *std::min_element(runtimes.begin(), runtimes.end());
        float _p99 = percentile(runtimes, 99.0);

        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Benchmark results (" << ITERS << " iterations)" << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "Average execution time: " << _mean << "ms" << std::endl;
        std::cout << "Standard deviation: " << _std << std::endl;
        std::cout << "Coefficient of variation: " << (_std / _mean) * 100 << "%" << std::endl;
        std::cout << "Minimum time: " << _min << "ms" << std::endl;
        std::cout << "Median: " << _median << "ms" << std::endl;
        std::cout << "P_99: " << _p99 << "ms" << std::endl;
        std::cout << std::endl;
    }

}

#endif // BENCHMARK_COMMON_CUH