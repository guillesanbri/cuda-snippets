#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils/stb_image.h"
#include "utils/stb_image_write.h"

using namespace std;

__global__
void rbgToGrayKernel(unsigned char *in, unsigned char *out, int w, int h, int c){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < w && y < h){
        int idxGray = y * w + x;
        int idxRGB = c * idxGray;
        unsigned char r = in[idxRGB];
        unsigned char g = in[idxRGB + 1];
        unsigned char b = in[idxRGB + 2];
        out[idxGray] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

void convertToGray(unsigned char *h_in, unsigned char *h_out, int w, int h, int c){
    // Allocate memory
    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, w * h * c);
    cudaMalloc((void **)&d_out, w * h);

    // Copy data
    cudaMemcpy(d_in, h_in, h * w * c, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);
    rbgToGrayKernel<<<gridSize, blockSize>>>(d_in, d_out, w, h, c);

    // Copy output data
    cudaMemcpy(h_out, d_out, (w * h), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_in); cudaFree(d_out);
}

int main(int argc, char **argv){
    int width, height, channels;

    // Load image using stb_image
    unsigned char *img = stbi_load("input.jpg", &width, &height, &channels, 3);
    if (!img){
        cerr << "Couldn't load the image." << endl;
        return -1;
    }

    // Output buffer
    // unsigned char *out = (unsigned char *)malloc(width * height);
    vector<unsigned char> out(width * height);

    convertToGray(img, out.data(), width, height, channels);

    stbi_write_jpg("output.jpg", width, height, 1, out.data(), 100);

    stbi_image_free(img);
    //free(out);

    return 0;
}