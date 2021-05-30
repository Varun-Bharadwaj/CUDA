#pragma once
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1
#define SIZE N*1024*1024
#define VectorSizeInBytes (SIZE*sizeof(float))


__host__ void VectorOperations(float* h_C);
__global__ void VectorInitializeWithWD(float* g_C);
__global__ void VectorInitializeAcrossWD(float* g_C);
__global__ void VectorInitializeCompiler(float* g_C);