#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    // Print some properties
    cout << "Max threads per block: " << devProp.maxThreadsPerBlock << endl;
    cout << "SM count: " << devProp.multiProcessorCount << endl;
    cout << "Clock Rate (kHz): " << devProp.clockRate << endl;
    cout << "Max threads per block (x): " << devProp.maxThreadsDim[0] << endl;
    cout << "Max threads per block (y): " << devProp.maxThreadsDim[1] << endl;
    cout << "Max threads per block (z): " << devProp.maxThreadsDim[2] << endl;
    cout << "Max blocks per grid (x): " << devProp.maxGridSize[0] << endl;
    cout << "Max blocks per grid (y): " << devProp.maxGridSize[1] << endl;
    cout << "Max blocks per grid (z): " << devProp.maxGridSize[2] << endl;
    cout << "Num. of threads in a warp: " << devProp.warpSize << endl;

    return 0;
}
