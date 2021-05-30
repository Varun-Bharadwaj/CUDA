#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MULTIPLIER 16
#define NUMBER_OF_POINTS MULTIPLIER*1024
#define NUMBER_OF_BYTES_POINTS NUMBER_OF_POINTS*sizeof(float3)
#define NUMBER_OF_BYTES_INDICES NUMBER_OF_POINTS*sizeof(int)

//CPU Function to compute the nearest neighbors of each point in 3D space.
void GenerateRandom3DPoints(float3* random3DPoints, int PointCount);
void VerifyResults(int* CPU_ClosestPointIndex, int* GPU_ClosestPointIndex, int PointCount);
void PrintNearestPoint(int* ClosestPointIndex, int PointCount);
void CPU_FindNearestNeighbors(float3* Points, int* ClosestPointIndex, int PointCount);

//GPU Naive Implementation using only Global Memory
__host__ void GPUGlobalNearestNeighbors(float3 *Points, int *ClosestPointIndex, int PointCount);
__global__ void gpuGlobalNN(float3* g_Points, int* g_ClosestPointIndex, int count);

//GPU Implementation using Global and Shared Memory - Extra Credit
__host__ void GPUSharedNearestNeighbors(float3* Points, int* ClosestPointIndex, int PointCount);
__global__ void gpuSharedNN(float3* g_Points, int* g_ClosestPointIndex, int count);


