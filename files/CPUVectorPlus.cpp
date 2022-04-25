#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define NB 1024

int N = 1 << 20;
int bytes = N * sizeof(int);
int* a = (int*)malloc(bytes);
int* b = (int*)malloc(bytes);
int* c = (int*)malloc(bytes);

void CPUVectorPlus(int start, int end) {
    for (int i = start; i < end; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    // 初始化数据
    for (int i = 0; i < N; i++) {
        a[i] = rand() * 100;
        b[i] = rand() * 100;
    }
    thread* threads;
    threads = (thread*)malloc(256 * sizeof(thread));

    int i = -1;
    int start = 0;
    int end = N / NB;
    while (end < N) {
        threads[++i] = thread(CPUVectorPlus, start, end);
        start = end; end += N / 256;
    }
    
    for (int i = 0; i < NB; i++) {
        threads[i].join();
    }

    for (int i = 0; i < N; i++) {
        int result = a[i] + b[i];
        if (result != c[i]) {
            printf("failed!\n");
            return -1;
        }
    }
    printf("Success!\n");
    return 0;

}