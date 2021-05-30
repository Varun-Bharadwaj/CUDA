#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define x_min 1.0 //The minimum value of x
#define x_max 2.0 // The maximum value of x

#define S 128
#define SIZE S*1024 // Number of x values
#define VECTOR_SIZE_IN_BYTES (SIZE*sizeof(float)) // x values are single precision numbers

#define M 300 // Precision Control Factor

//Utility Functions
	//Funtion Prototype to generate x data
	void OnGenerateX(float* xTemp);
	//Function generate the ln(x) using the math library functions
	void OnGenerateRef(float* xTemp, float* h_NLTemp, int nSizeTemp);
	//function to verify results by inspection
	void OnPrintResults(float* xTemp, float* RefNL, float* CPUNaiveNLTemp, float* CPUOptNL, float* GPUNL, int nSize);

//CPU Functions
	//function prototype to compute ln(x) on the CPU Naively
	void OnCPUNaiveComputeNL(float* xTemp, float* NLTemp, int nSize, int precFactor);

	//function prototype to compute ln(x) on the CPU Optimized
	void OnCPUOptimizedComputeNL(float* xTemp, float* NLTemp, int nSize, int precFactor);

//GPU Functions
	//GPU Host Function
	__host__ void OnGPUComputeNL(float* h_xTemp, float* h_NLTemp, int nSizeTemp, int MTemp);

	//GPU Kernel Function
	__global__ void OnComputeNL(float* g_xTemp, float* g_NLTemp, int nSize, int precFactor);
	

