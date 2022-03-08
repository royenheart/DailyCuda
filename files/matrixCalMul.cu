#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct matrix {
    int width;
    int height;
    double *elements;
}matrix;

// 返回矩阵对应位置元素
__device__ double getMatrixElements(matrix *a, int row, int col) {
    return a->elements[row * a->width + col];
}

// 设置矩阵对应位置元素
__device__ void setMatrixElements(matrix *a, int row, int col, double value) {
    a->elements[row * a->width + col] = value;
}

// 进行矩阵乘法运算
__global__ void matrixMul(matrix *a, matrix *b, matrix *c) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    double value = 0;
    for (int i = 0; i < a->width; i++) {
        value += getMatrixElements(a, row, i) * getMatrixElements(b, i, col);
    }
    setMatrixElements(c, row, col, value);
}

int main(int argc, char *args) {
    // 指定矩阵大小
    int widthA = 1 << 10;
    int heightA = 1 << 8;
    int widthB = 1 << 8;
    int heightB = 1 << 10;

    // 指定矩阵占用字节数
    int bytesA = heightA * widthA * sizeof(double);
    int bytesB = heightB * widthB * sizeof(double);
    int bytesC = heightA * widthB * sizeof(double);

    // int *eleFinal;
    matrix *a, *b, *c;
    // matrix *result;

    // cudaMallocManaged((void**)&eleFinal, bytesC);
    // cudaMallocManaged((void**)&result, sizeof(matrix));
    cudaMallocManaged((void**)&a, sizeof(matrix));
    cudaMallocManaged((void**)&b, sizeof(matrix));
    cudaMallocManaged((void**)&c, sizeof(matrix));
    // 分配内存空间
    cudaMallocManaged((void**)&a->elements, bytesA);
    cudaMallocManaged((void**)&b->elements, bytesB);
    cudaMallocManaged((void**)&c->elements, bytesC);
    
    // 初始化数据

    a->width = widthA;
    a->height = heightA;
    b->width = widthB;
    b->height = heightB;
    c->width = widthB;
    c->height = heightA;
    // result->width = widthB;
    // result->height = heightA;

    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthA; j++) {
            a->elements[i * widthA + j] = 1.0;
        }
    }
    
    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
            b->elements[i * widthB + j] = 2.0;
        }
    }

    memset(c->elements, 0, bytesC);
    // result->elements = eleFinal;

    // grid和block块大小划分
    dim3 blockSize(32, 32);
    dim3 gridSize((widthB + blockSize.x - 1) / blockSize.x, (heightA + blockSize.y - 1) / blockSize.y);
    
    // 执行核函数
    matrixMul<<<gridSize, blockSize>>>(a, b, c);

    // 同步device和host
    cudaDeviceSynchronize();

    // 验证
    int isRight = 1;
    for (int row = 0; row < heightA && isRight == 1; row++) {
        for (int col = 0; col < widthB && isRight == 1; col++) {
            double value = 0;
            for (int i = 0; i < widthA; i++) {
                value += a->elements[row * widthA + i] * b->elements[i * widthB + col];
            }
            if (value != c->elements[row * widthB + col]) {isRight = 0;}
        }
    }
    printf("%s\n", (isRight == 1)?"Results are fine!":"False!");

    // 释放内存
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
} 