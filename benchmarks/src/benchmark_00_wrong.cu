// 
// This code is very wrong! Don't measure anything doing this.
//
#include <chrono>
#include <iostream>
#include <vector>
#include "benchmark_common.cuh"

using namespace std;

void benchmarkWrong(float *h_A, float *h_B, float *h_C, int m, int k, int n){
    // Allocate GPU memory
    float *d_A;
    float *d_B;
    float *d_C;
    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy input data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Start measuring time (wrong!)
    auto start = chrono::steady_clock::now();

    // Launch kernel
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y, 1); 
    benchmark::matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);

    // Simulate CPU work

    // Finish measuring time (wrong!)
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Kernel took: " << duration.count() << "us" << endl;

    // Copy output to host
    // cudaMemcpy(...) synchronizes indirectly, for demonstration purposes 
    // leave it outside of the benchmark to make it even worse
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char **argv){
    if (argc < 2){
        cerr << "Usage: " << argv[0] << " <MATRIX_SIZE>" << endl;
        return 1;
    }

    int MATRIX_SIZE;
    try {
        MATRIX_SIZE = stoi(argv[1]);
        if (MATRIX_SIZE <= 0) throw invalid_argument("Matrix size must be positive.");
    }
    catch (const exception &e) {
        cerr << "Error: Invalid matrix size: " << e.what() << endl;
        return 1;
    }
    
    // Initialise random matrices
    vector<float> A(MATRIX_SIZE * MATRIX_SIZE);
    vector<float> B(MATRIX_SIZE * MATRIX_SIZE);
    vector<float> C(MATRIX_SIZE * MATRIX_SIZE);
    benchmark::randomizeVector(A);
    benchmark::randomizeVector(B);

    // Run the benchmark
    benchmarkWrong(A.data(), B.data(), C.data(), MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);

    return 0;
}