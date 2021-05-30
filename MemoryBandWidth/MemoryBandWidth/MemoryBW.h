#pragma once
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE_MAX 1.0
#define RANGE_MIN -1.0


void InitializeVector(float* vect, const int nSize);

//GPU Helper Function
__host__ void MemoryBenchMarks(float a,float *x,float *y,int nSize);

//GPU Kernel Functions to measure the global memory bandwidth
__global__ void saxpyGM(float g_a,float* g_x, float* g_y, int nSize);