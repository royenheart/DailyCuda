#ifndef __CUDATOOLS_H__
#define __CUDATOOLS_H__
#endif

#include <time.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

#ifdef _WIN32
int gettimeofday(struct timeval *tp) {
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds * 1000;
	return 0;
}
#endif

double getExecuteTime() {
    struct timeval te;
    gettimeofday(&te);
    return (double)te.tv_sec + (double)te.tv_usec*1e-6;
}

void HANDLE_ERROR(cudaError_t error) {
	printf("Cuda Error happened!, error code: %d\n", error);
	printf("Please check the website https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038 to see what happened\n");
}