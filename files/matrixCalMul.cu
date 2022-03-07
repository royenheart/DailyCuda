/* 矩阵乘法 */

/**
 * @author RoyenHeart
 * @since 2021.09.25
 */

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "cudaTools.h"

using namespace std;

__global__ void matrixCalMul(float *A, float *B, float *C) {
    int ic = blockDim.x * blockIdx.x + threadIdx.y; // 定位C集合元素
    int ib = blockDim.y * blockIdx.y + threadIdx.y; // 定位B集合元素
    int ia = blockDim.x * blockIdx.x + threadIdx.x; // 定位A集合元素
    
    C[ic] += A[ia]*B[ib];
}

int main(int argv, char **argc) {

    /* 声明矩阵大小 */
    int xA = 1<<4;
    int yA = 1<<8;
    int xB = 1<<8;
    int yB = 1<<6;
    int tA = xA * yA;
    int tB = xB * yB;
    size_t sizeA = sizeof(float)*tA;
    size_t sizeB = sizeof(float)*tB;
    int tC = yA * xB;
    size_t sizeC = sizeof(float)*tC;

    /* 分配主机内存 */
    float *mA = (float*)malloc(sizeA);
    float *mB = (float*)malloc(sizeB);
    float *mC = (float*)malloc(sizeB);
    
    if (mA == NULL || mB == NULL || mC == NULL) {
        fprintf(stderr,"failed to allocate memory for host\n");
        exit(EXIT_FAILURE);
    }
    
    /* 随机生成数据 */
    for (int i = 0; i < tA; i++) {
        mA[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < tB; i++) {
        mB[i] = (float)rand() / (float)RAND_MAX;
    }

    /* 分配设备内存 */
    float *GmA = NULL;
    float *GmB = NULL;
    float *GmC = NULL;
    cudaMalloc(&GmA,sizeA);
    cudaMalloc(&GmB,sizeB);
    cudaMalloc(&GmC,sizeC);

    /* 主机数据转移至设备 */
    cudaMemcpy(GmA,mA,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(GmB,mB,sizeB,cudaMemcpyHostToDevice);

    /* 声明线程块和线程空间 */
    dim3 blocksPerGrid(yA,yA);
    dim3 threadsPerBlock(xB,xB);

    /* 启动核函数计算 */
    double start,end;
    start = getExecuteTime();
    matrixCalMul<<<blocksPerGrid,threadsPerBlock>>>(GmA,GmB,GmC);
    end = getExecuteTime();
    printf("Total execute time on GPU is %lfs\n",end-start);

    /* 设备数据转移至主机 */
    cudaMemcpy(mC,GmC,sizeC,cudaMemcpyDeviceToHost);

    /* 释放主机内存空间 */
    free(mA);
    free(mB);
    free(mC);

    /* 释放设备内存空间 */
    cudaFree(GmA);
    cudaFree(GmB);
    cudaFree(GmC);

    return 0;
}