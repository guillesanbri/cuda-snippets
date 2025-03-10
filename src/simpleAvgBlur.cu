#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils/stb_image.h"
#include "utils/stb_image_write.h"

using namespace std;

const int BLUR_SIZE=3;

__global__
void simpleAvgBlurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if ( x < w && y < w ){
        int px_val = 0;
        int px_count = 0;
        for ( int blur_x = x - BLUR_SIZE; blur_x < x + BLUR_SIZE + 1; blur_x++ ){
            for ( int blur_y = y - BLUR_SIZE; blur_y < y + BLUR_SIZE + 1; blur_y++ ){
                if ( blur_x >= 0 && blur_x < w && blur_y >= 0 && blur_y < h ){
                    px_val += in[blur_y * w + blur_x];
                    px_count++;
                }
            }
        }
        out[y * w + x] = (unsigned char)(px_val / px_count);
    }

}

void convertToGray(unsigned char *h_in, unsigned char *h_out, int w, int h){
    // Allocate memory
    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, w * h);
    cudaMalloc((void **)&d_out, w * h);

    // Copy data
    cudaMemcpy(d_in, h_in, h * w, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);
    simpleAvgBlurKernel<<<gridSize, blockSize>>>(d_in, d_out, w, h);

    // Copy output data
    cudaMemcpy(h_out, d_out, (w * h), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_in); cudaFree(d_out);
}

int main(int argc, char **argv){
    int width, height, channels;

    // Load image using stb_image
    unsigned char *img = stbi_load("input.jpg", &width, &height, &channels, 1);
    if (!img){
        cerr << "Couldn't load the image." << endl;
        return -1;
    }

    // Output buffer
    vector<unsigned char> out(width * height);

    convertToGray(img, out.data(), width, height);

    stbi_write_jpg("output.jpg", width, height, 1, out.data(), 100);

    stbi_image_free(img);

    return 0;
}