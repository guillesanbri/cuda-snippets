#include <stdio.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n){
    
    // Allocate memory
    float *d_A, *d_B, *d_C;
    int size = n * sizeof(float);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vecAddKernel<<<ceil(n / 256.0), 256>>>(d_A, d_B, d_C, n);

    // Copy output
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}

void printArraySummary(float* a, int n, int show=3){
    printf("[");
    for (int i = 0; i < show && i < n; i++){
        printf("%.1f", a[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > 2 * show) printf(", ..., ");
    int start = (n > 2 * show) ? (n - show) : show;
    for (int i = start; i < n; i++){
        printf("%.1f", a[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    
    // Initialize sample data
    int n = 300;
    int size = n * sizeof(float);

    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);
    
    for (int i = 0; i < n; i++){
        a[i] = i*10;
        b[i] = i*20;
    }

    // Compute
    vecAdd(a, b, c, n);

    // Inspect results
    printArraySummary(c, n);

    // Free host memory
    free(a); free(b); free(c);

    return 0;
}
