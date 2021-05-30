#include "MemoryBW.h"

//GPU Kernel Functions to measure the global memory bandwidth
__global__ void saxpyGM(float g_a, float* g_x, float* g_y, int nSize)
{
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < nSize)
	{
		g_y[idx] = g_a * g_x[idx] + g_y[idx];
	}
}

//GPU Helper Function
__host__ void MemoryBenchMarks(float a, float* x, float* y, int N)
{
	float* d_x{}, * d_y{};
	//chrono::time_point<std::chrono::system_clock> start, end;
	cudaEvent_t start, end;
	int VECTOR_SIZE_IN_BYTES = N * sizeof(float);

	//Allocating Memory
	cudaError_t cudaStatus = cudaMalloc((void**)&d_x, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess)
	{
		cout << "dev_a: cudaMalloc Failed" << endl;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_y, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess)
	{
		cout << "dev_a: cudaMalloc Failed" << endl;
		goto Error;
	}
	
	//Copy data on the host to the device using the cudaMemcpy function
	cudaStatus = cudaMemcpy(d_x, x, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "h_x to d_x: cudaMemcpy Failed" << endl;
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_y, y, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "h_y to d_y: cudaMemcpy Failed" << endl;
		goto Error;
	}

	//Creating number of threads/block and number of blocks
	int threads_PER_BLOCK = 1024;
	int blocks_PER_GRID = (int)ceil(N / threads_PER_BLOCK);
	cout << "Vector Size = " << N << endl;
	cout << "Number of Threads/Block = " << threads_PER_BLOCK << endl;
	cout << "Number of Blocks/Grid = " << blocks_PER_GRID << endl;

	//Instantiate a object
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	//Launch the kernel on the GPU and continue
	//start = std::chrono::system_clock::now();
	cudaEventRecord(start);
	saxpyGM << <blocks_PER_GRID, threads_PER_BLOCK >> > (a, d_x, d_y,N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "SAXPY Kernel launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		goto Error;
	}
	//Wait for the kernel to finish 
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	cout << "cudaDeviceSynchronize error with error code" << cudaStatus << endl;
	//	goto Error;
	//}
	//end = std::chrono::system_clock::now();
	//std::chrono::duration<double> elasped_seconds = end - start;
	//Compute the Global Memory Bandwidth
	float memoryTransferTime = 0.0;
	cudaEventElapsedTime(&memoryTransferTime, start, end);
	//memoryTransferTime = elasped_seconds.count();
	float EffectiveBW = (N * sizeof(float) * 3) / memoryTransferTime;
	cout << "Effective BW = " << (EffectiveBW / 1e6) << " GB/s" << endl;

Error:
	cudaFree(d_x);
	cudaFree(d_y);
	//Reset the device before exiting for profiler tools like Nsight and Visual Profiler to show complete traces
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cout << "cudaDeviceReset Failed" << endl;
	}
}

