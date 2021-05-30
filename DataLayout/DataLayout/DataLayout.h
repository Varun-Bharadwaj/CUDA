#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LEN 1024*1024

//Array of Structures
struct AoS {
	float x;
	float y;
};

void InitializeAoS(AoS* aos, int size);
void cpuComputeAoS(AoS* in, AoS* out, int size);

//Structure of Arrays
struct SoA {
	float x[LEN];
	float y[LEN];
};

void Copy_AoSToSoA(AoS* aos, SoA* soa, int size);
void cpuComputeSoA(SoA* in, SoA* out, int size);

//Verification
void VerifyOnCPU(AoS* aos, SoA* soa, int size);


//GPU Host Functions and Kernels

__host__ void GPUAoS(AoS* h_in, AoS* h_out, int size);
__global__ void gpuComputeAoS(AoS* g_in, AoS* g_out, int n);
void VerifyGPUAoS(AoS* cpu_aos, AoS* gpu_aos, int size);

__host__ void GPUSoA(SoA* h_in, SoA* h_out, int size);
__global__ void gpuComputeSoA(SoA* g_in, SoA* g_out, int n);
void VerifyGPUSoA(SoA* cpu_soa, SoA* gpu_soa, int size);