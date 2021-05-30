#include "MemComparison.h"

__global__ void GlobalBM(int* g_Vector, int* g_Result, int size, int count)
{
	for (int i = 0; i < count; i++)
	{
		g_Result[i] = 0;
		for (int j = 0; j < size; j++)
		{
			g_Result[i] += g_Vector[j];
		}
	}
}

__global__ void RegisterBM(int* g_Vector, int* g_Result, int size, int count)
{0;
int temp;
	for (int i = 0; i < count; i++)
	{
		temp = 0;
		g_Result[i] = 0;
		for (int j = 0; j < size; j++)
		{
			temp += g_Vector[j];
		}
		g_Result[i] += temp;
	}
}

__host__ void MemoryBenchmark(int* pVector, int* pResult)
{
	int* d_Vector, * d_Result;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;

	//Create Event Objects
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);

	cudaMalloc((void**)&d_Vector, DATA_SIZE_BYTES);
	cudaMalloc((void**)&d_Result, DATA_SIZE_BYTES);

	cudaMemcpy(d_Vector, pVector, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);

	//Global Memory
	cudaEventRecord(kernel_start);
	//Launch the GlobalMemory Kernel
	GlobalBM << <1, 1 >> > (d_Vector, d_Result, SIZE, ITERATION_COUNT);
	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "Global Memory Kernel launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop);
	cout << "Global Memory Use: Elapsed Time (GPU) = " << fElapsedTime << " msecs" << endl;
	cudaMemcpy(pResult, d_Result, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_Vector);
	cudaFree(d_Result);
	

	cudaMalloc((void**)&d_Vector, DATA_SIZE_BYTES);
	cudaMalloc((void**)&d_Result, DATA_SIZE_BYTES);

	cudaMemcpy(d_Vector, pVector, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	

	//Register Memory
	cudaEventRecord(kernel_start);
	//Launch the Register Memory Kernel
	RegisterBM << <1, 1 >> > (d_Vector, d_Result, SIZE, ITERATION_COUNT);
	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "Register Memory Kernel launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop);
	cout << "Register Memory Use: Elapsed Time (GPU) = " << fElapsedTime << " msecs" << endl;
	cudaMemcpy(pResult, d_Result, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_Vector);
	cudaFree(d_Result);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaDeviceReset();
}