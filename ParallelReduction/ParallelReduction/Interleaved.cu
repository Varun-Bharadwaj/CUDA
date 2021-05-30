#include "ParallelReduction.h"
#include "GPUErrors.h"

__global__ void OnInterleavedWithoutUnroll(float* g_Vector, float* g_PartialSum)
{
	//Save threadIdx.x on the register
	int tid = threadIdx.x;

	//Compute the global thread index
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);

	if (idx >= VECTOR_SIZE)
	{
		return;
	}

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		g_PartialSum[blockIdx.x] = blockAddress[0];
	}
}

__host__ void OnInterleaved(float* vectorTemp)
{
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	chrono::time_point<std::chrono::system_clock> start, end;

	float fElapsedTime;
	float fPartialReduceTime;

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	float* d_Vector;
	float* d_PartialSum;

	float* h_PartialSum;

	//Block and Thread Parameters
	dim3 block(256);
	dim3 grid((VECTOR_SIZE + block.x - 1) / block.x, 1);
	cout << "Neighborhood Implementations" << endl;
	cout << "\tThreads/Block: " << block.x << endl;
	cout << "\tBlocks/Grid: " << grid.x << endl;

	//The partial sums of each block
	h_PartialSum = new float[grid.x];

	//Allocate memory on the GPU to store the vector and partial sums
	HandleCUDAError(cudaMalloc((void**)&d_Vector, VECTOR_SIZE_IN_BYTES));
	HandleCUDAError(cudaMalloc((void**)&d_PartialSum, (grid.x * sizeof(float))));

	//Copy the vector to the GPU from the host
	HandleCUDAError(cudaMemcpy(d_Vector, vectorTemp, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));

	//Launch the Interleaved pairing kernel without unrolling
	HandleCUDAError(cudaEventRecord(kernel_start));
	OnInterleavedWithoutUnroll << <grid, block >> > (d_Vector, d_PartialSum);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fPartialReduceTime, kernel_start, kernel_stop));

	//Copy the vector to the GPU from the host containing the sum of each block
	HandleCUDAError(cudaMemcpy(h_PartialSum, d_PartialSum, (grid.x * sizeof(float)), cudaMemcpyDeviceToHost));

	float sum = 0.0f;
	start = std::chrono::system_clock::now();
	for (int j = 0; j < grid.x; j++)
	{
		sum += h_PartialSum[j];
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	fElapsedTime = fPartialReduceTime + (elasped_seconds.count() * 1000.0f);
	cout << "GPU Interleaved Reduction without unroll Execution time: " << fElapsedTime << " msecs" << endl;
	cout << "\t\tGPU Interleaved Reduction: " << sum << endl;

	HandleCUDAError(cudaFree(d_Vector));
	HandleCUDAError(cudaFree(d_PartialSum));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}