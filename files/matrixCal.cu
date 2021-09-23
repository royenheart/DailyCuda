/* 矩阵运算 */

/**
 * @author RoyenHeart
 * @since 2021.09.22
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cudaTools.h"

#define SIZEONE 1<<8
#define FAILEDALLOCATE "failed to allocate memory fot device\n"
#define FAILEDTRANSFER "failed to transfer memory between two\n"

struct ErrPool {
    private:cudaError_t err[SIZEONE] = {cudaSuccess};
    private:int top;
    public:ErrPool() {
        top = -1;
    }

    public:int addErr(cudaError_t type, const char *possibleErr) {
        if (top == SIZEONE - 1) {
            top = -1;
        }
        err[++top] = type;
        if(err[top] != cudaSuccess) {
            fprintf(stderr, possibleErr);
            exit(EXIT_FAILURE);
        }
        return 0;
    }
};

__global__ void matrixCalAdd(float *A, float *B, float *C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // 定位元素，使用一维
    
    C[i] = A[i] + B[i];
}

int main() {
    ErrPool errPool;

    // 声明矩阵大小
    int n = 1<<12;
    int size = n*n;
    size_t actualSize = size * sizeof(float);

    // 分配主机内存
    float *matrixCpuA = (float*)malloc(actualSize);
    float *matrixCpuB = (float*)malloc(actualSize);
    float *matrixCpuC = (float*)malloc(actualSize);
    
    // 判断是否正确分配主机内存
    if (matrixCpuA == NULL || matrixCpuB == NULL || matrixCpuC == NULL) {
        fprintf(stderr, "failed to allocate host memory");
        exit(EXIT_FAILURE);
    }

    // 初始化数据
    for (int i = 0; i < size; i++) {
        matrixCpuA[i] = (float)rand() / (float)RAND_MAX;
        matrixCpuB[i] = (float)rand() / (float)RAND_MAX;
    }

    // 分配设备内存
    float *matrixGpuD = NULL;
    float *matrixGpuE = NULL;
    float *matrixGpuF = NULL;
    errPool.addErr(cudaMalloc(&matrixGpuD, actualSize),FAILEDALLOCATE);
    errPool.addErr(cudaMalloc(&matrixGpuE, actualSize),FAILEDALLOCATE);
    errPool.addErr(cudaMalloc(&matrixGpuF, actualSize),FAILEDALLOCATE);

    // 主机数据转移至设备
    errPool.addErr(cudaMemcpy(matrixGpuD, matrixCpuA, actualSize, cudaMemcpyHostToDevice),FAILEDTRANSFER);
    errPool.addErr(cudaMemcpy(matrixGpuE, matrixCpuB, actualSize, cudaMemcpyHostToDevice),FAILEDTRANSFER);

    // 声明使用的线程/线程块和线程块数6
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // 从CPU调用核函数
    double start,end;
    start = getExecuteTime();
    matrixCalAdd<<<blocks,threadsPerBlock>>>(matrixGpuD,matrixGpuE,matrixGpuF);
    end = getExecuteTime();
    printf("Total execute time on GPU is %lfs\n",end-start);
    
    // 核函数计算完毕后从设备内存转移数据至主机
    errPool.addErr(cudaMemcpy(matrixCpuC, matrixGpuF, actualSize, cudaMemcpyDeviceToHost),FAILEDTRANSFER);

    // 检查错误，精度要求为1e-5
    for (int i = 0; i < size; i++) {
        if (fabs(matrixCpuA[i] + matrixCpuB[i] - matrixCpuC[i]) > 1e-5) {
            fprintf(stderr,"Answer Wrong!\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("PASSWED!\n");

    // 释放主机内存
    free(matrixCpuA);
    free(matrixCpuB);
    free(matrixCpuC);

    // 释放设备内存
    cudaFree(matrixGpuD);
    cudaFree(matrixGpuE);
    cudaFree(matrixGpuF);

    return 0;
}