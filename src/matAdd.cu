#include <iostream>
#include <vector>

using namespace std;

__global__
void simpleMatAddKernel(float *A, float *B, float *C, int side){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * side + x;
    C[idx] = A[idx] + B[idx]; 
}

void matAdd(float *h_A, float *h_B, float *h_C, int side){
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    int matrix_size = side * side * sizeof(float);
    cudaMalloc((void **)&d_A, matrix_size);
    cudaMalloc((void **)&d_B, matrix_size);
    cudaMalloc((void **)&d_C, matrix_size);

    // Copy input data to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((side + blockDim.x - 1 )/blockDim.x, (side + blockDim.y - 1 )/blockDim.y, 1);
    // TODO: Measure performance
    // TODO: Include one-thread-per-row approach
    // TODO: Include one-thread-per-col approach
    simpleMatAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, side);

    // Copy output to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(int argc, char **argv){
    // Define input vectors
    const int SIDE=100;
    vector<float> h_A(SIDE * SIDE, 10.0f);
    vector<float> h_B(SIDE * SIDE, 20.0f);
    vector<float> h_C(SIDE * SIDE, 0.0f);

    // Initialise values for A and B
    for (size_t i=0; i < (SIDE * SIDE); i++){
        // TODO: Generate random numbers here
    }

    // Launch matrix addition
    matAdd(h_A.data(), h_B.data(), h_C.data(), SIDE);

    cout << h_C[0] << endl;

    return 0;
}