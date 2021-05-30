#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE_MAX 1.0
#define RANGE_MIN -1.0

//HOST Functions
void InitializeMatrix(float* temp, const int ny, const int nx);
void ZeroMatrix(float* temp, const int ny, const int nx);
void MatrixAdditionOnHost(float* A, float* B, float* C, const int ny, const int nx);
void MatrixAdditionVerification(float* hostC, float* gpuC, const int ny, const int nx);
void DisplayMatrix(string name, float* temp, const int ny, const int nx);


//GPU Host Function
__host__ void MatrixAdditionOnGPU2DG2DB(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx);
__host__ void MatrixAdditionOnGPU2DG1DB(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx);
__host__ void MatrixAdditionOnGPU1DG1DB(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx);

//GPU Kernel Functions
__global__ void MatrixAddition2DG2DB(float* g_A, float* g_B, float* g_C, const int ny, const int nx);
__global__ void MatrixAddition2DG1DB(float* g_A, float* g_B, float* g_C, const int ny, const int nx);
__global__ void MatrixAddition1DG1DB(float* g_A, float* g_B, float* g_C, const int ny, const int nx);
