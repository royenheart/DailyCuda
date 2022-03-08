#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void VectorPlus(int *a, int *b, int *c, int n) {
    // int thread = threadIdx.x + blockIdx.x * blockDim.x;
    // int foot = gridDim.x * blockDim.x;

    // for (int i = thread; i < n; i+= foot) {
        // c[i] = a[i] + b[i];
    // }

    // 不使用步长
    int number = threadIdx.x + blockIdx.x * blockDim.x;

    c[number] = a[number] + b[number];
}

int main(int argc, char *args[]) {

    // 分配host内存
    int N = 1 << 20;
    int bytes = N * sizeof(int);
    int *a = (int*)malloc(bytes);
    int *b = (int*)malloc(bytes);
    int *c = (int*)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        a[i] = rand() * 100;
        b[i] = rand() * 100;
    }

    // 申请device内存
    int *d_a;
    int *d_b;
    int *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);
    
    // 拷贝host数据到device
    cudaMemcpyAsync((void*)d_a, (void*)a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync((void*)d_b, (void*)b, bytes, cudaMemcpyHostToDevice);
    
    // 进行kernel配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // 执行kernel函数
    VectorPlus<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 将device数据拷贝至host
    cudaMemcpyAsync((void*)c, (void*)d_c, bytes, cudaMemcpyDeviceToHost);

    // 数据监测
    int isRight = 1;
    for (int i = 0; i < N; i++) {
        int result = a[i] + b[i];
        if (result != c[i]) {isRight = 0;break;}
    }
    printf("%s\n", (isRight == 1)?"Results are fine!":"False!");

    // 释放内存数据
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}