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
        printf("Amounts of SM: %d\n", DevProp.multiProcessorCount);
        printf("Shared cache per block: %zd kb\n", DevProp.sharedMemPerBlock);
        printf("Max processors per block: %d\n", DevProp.maxThreadsPerBlock);
        printf("Max threads per EM: %d\n", DevProp.maxThreadsPerMultiProcessor);
        printf("Max threads per SM: %d\n", DevProp.maxThreadsPerMultiProcessor / 32);
    }

    return 0;
}