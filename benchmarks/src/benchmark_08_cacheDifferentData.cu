#include <chrono>
#include <iostream>
#include <vector>
#include "benchmark_common.cuh"

using namespace std;

void benchmarkCacheDifferentData(vector<float> A, vector<float> B, vector<float> C, int m, int k, int n){

    float *h_A = A.data();
    float *h_B = B.data();
    float *h_C = C.data();

    // Allocate GPU memory
    float *d_A;
    float *d_B;
    float *d_C;
    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, sizeC));

    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Define block and grid
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y, 1); 

    // Launch the kernel a few times to avoid cold start
    for (int i=0; i < benchmark::CS_ITERS; i++){
        benchmark::matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
        CHECK_LAST_CUDA_ERROR();
    }

    // Randomize the matrices again
    benchmark::randomizeVector(A);
    benchmark::randomizeVector(B);
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Results vector
    vector<float> runtimes(benchmark::ITERS);

    for (int i=0; i<benchmark::ITERS; i++){

        // Start measuring time
        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        // Launch kernel
        benchmark::matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
        // Check for sync errors in the kernel launch
        CHECK_LAST_CUDA_ERROR();
        // Stop measuring time
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        // Compute time
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Store the measurement
        runtimes[i] = milliseconds;
    }

    // Print benchmark info
    benchmark::printResults(runtimes);

    // Copy output to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
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
    benchmarkCacheDifferentData(A, B, C, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);

    return 0;
}