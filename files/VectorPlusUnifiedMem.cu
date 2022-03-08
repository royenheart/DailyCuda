#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void VectorPlusUnified(int *a, int *b, int *c, int N) {
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    c[thread] = a[thread] + b[thread];
}

int main() {
    int N = 1 << 20;
    int bytes = N * sizeof(int);
    
    int *a, *b, *c, *result; 
    cudaMallocManaged((void**)&a, bytes);
    cudaMallocManaged((void**)&b, bytes);
    cudaMallocManaged((void**)&c, bytes);
    cudaMallocManaged((void**)&result, bytes);

    for (int i = 0; i < N; i++) {
        a[i] = rand() * 100;
        b[i] = rand() * 100;
    }

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    VectorPlusUnified<<<gridSize, blockSize>>>(a, b, c, N);

    cudaDeviceSynchronize();

    int isRight = 1;
    for (int i = 0; i < N; i++) {
        result[i] = a[i] + b[i];
        if (result[i] != c[i]){isRight = 0;break;}
    }
    printf("%s\n", (isRight == 1)?"Results are fine!":"False!");

    // 释放内存数据
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    return 0;
}