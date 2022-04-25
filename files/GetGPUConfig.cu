#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaTools.h"

// 获取GPU信息
int main(char* args[], int argc) {
    // 设置cudaDeviceProp结构读取GPU信息
    cudaDeviceProp DevProp;
    
    int count;

    cudaGetDeviceCount(&count);
    printf("Amounts of GPU: %d\n", count);
    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&DevProp, i);
        printf("GPU Device: %d: %s\n", i, DevProp.name);
        printf("GPU major compute capability: %d\n", DevProp.major);
        printf("GPU minor compute capability: %d\n", DevProp.minor);
        printf("GPU Total Global Memory: %zd\n", DevProp.totalGlobalMem);
        printf("Warp size in threads: %d\n", DevProp.warpSize);
        printf("Amounts of SM: %d\n", DevProp.multiProcessorCount);
        printf("Shared cache per block: %zd kb\n", DevProp.sharedMemPerBlock);
        printf("Max threads per block: %d\n", DevProp.maxThreadsPerBlock);
        printf("Max threads on dimension x of a block: %d\n", DevProp.maxThreadsDim[0]);
        printf("Max threads on dimension y of a block: %d\n", DevProp.maxThreadsDim[1]);
        printf("Max threads on dimension z of a block: %d\n", DevProp.maxThreadsDim[2]);
        printf("Max threads on dimension x of a grid: %d\n", DevProp.maxGridSize[0]);
        printf("Max threads on dimension y of a grid: %d\n", DevProp.maxGridSize[1]);
        printf("Max threads on dimension z of a grid: %d\n", DevProp.maxGridSize[2]);
        printf("Max resident threads per Multi Processor: %d\n", DevProp.maxThreadsPerMultiProcessor);
        printf("Max resident blocks per Multi Processor: %d\n", DevProp.maxBlocksPerMultiProcessor);
    }

    return 0;
}