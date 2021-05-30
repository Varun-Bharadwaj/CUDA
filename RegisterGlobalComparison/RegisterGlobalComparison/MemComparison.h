#pragma once
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 31
#define DATA_SIZE_BYTES SIZE*sizeof(int)
#define ITERATION_COUNT 64*1024
#define RANGE_MIN 0
#define RANGE_MAX 30

void InitVector(int* Vector);

__host__ void MemoryBenchmark(int* pVector, int* pResult);
__global__ void GlobalBM(int* g_Vector, int* g_Result, int size, int count);
__global__ void RegisterBM(int* g_Vector, int* g_Result, int size, int count);