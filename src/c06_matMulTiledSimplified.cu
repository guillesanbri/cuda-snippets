// This kernel assumes that we are multiplying squared matrices, and that the width of the matrices is a multiple of the block size
#include <iostream>
#include <vector>

using namespace std;

const int width = 1024;
const int TILE_W = 16;

__global__
void matMulKernel(float *d_M, float *d_N, float *d_P, int width){
    // M * N = P
    __shared__ float tileM[TILE_W][TILE_W];
    __shared__ float tileN[TILE_W][TILE_W];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowP = by * blockDim.y + ty;
    int colP = bx * blockDim.x + tx;

    float pValue = 0;

    // Loop over the tiles
    for (int t=0; t < width / TILE_W; t++){
        // Each threads fetches one value from global memory
        tileM[ty][tx] = d_M[rowP * width + t * TILE_W + tx];
        tileN[ty][tx] = d_N[(t * TILE_W + ty) * width + colP];

        __syncthreads();

        for (int i=0; i<TILE_W; i++){
            pValue += tileM[ty][i] * tileN[i][tx];
        }

        __syncthreads();
    }

    d_P[rowP * width + colP] = pValue;
}

void matMul(float *h_M, float *h_N, float *h_P, int width){
    // Allocate memory
    float *d_N, *d_M, *d_P;
    int size = width * width * sizeof(float);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_P, size);

    // Copy input to device
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(TILE_W, TILE_W, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (width + blockDim.y - 1) / blockDim.y, 1);
    matMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, width);

    // Copy output to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_N); cudaFree(d_M); cudaFree(d_P);
}

int main(int argc, char **argv){

    // Define input matrices
    vector<float> M(width*width, 1);
    vector<float> N(width*width, 2);
    vector<float> P(width*width, 0);

    matMul(M.data(), N.data(), P.data(), width);

    cout << P[0] << endl;
    return 0;
}