#include "DataLayout.h"
#include "GPUErrors.h"

__host__ void GPUAoS(AoS* h_in, AoS* h_out, int size)
{
	AoS* d_in, * d_out;
	const int SizeInBytes_AoS = LEN * sizeof(AoS);
	cout << "\n\nGPU Execution using Array of Structures (AoS)" << endl;
	cout << "\tArray of Structures (AoS) Size in Bytes = " << SizeInBytes_AoS << " bytes" << endl;

	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((AoS**)&d_in, SizeInBytes_AoS));
	HandleCUDAError(cudaMalloc((AoS**)&d_out, SizeInBytes_AoS));

	//Transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaMemcpy(d_in, h_in, SizeInBytes_AoS, cudaMemcpyHostToDevice));

	//Kernel Invoke Parameters - 1D Grid and 1D Blocks
	int dimx = 256;
	dim3 block(dimx, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);

	cout << "\t1D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\t1D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;

	HandleCUDAError(cudaEventRecord(kernel_start));
	gpuComputeAoS << <grid, block >> > (d_in, d_out, size);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	HandleCUDAError(cudaMemcpy(h_out, d_out, SizeInBytes_AoS, cudaMemcpyDeviceToHost));
	cout << "\tGPU Computation using Array of Structures (AoS): Elapsed Time = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_in));
	HandleCUDAError(cudaFree(d_out));

	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}

__global__ void gpuComputeAoS(AoS* g_in, AoS* g_out, int n)
{
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < n)
	{
		AoS temp = g_in[idx];
		temp.x += 10.0f;
		temp.y += 20.0f;
		g_out[idx] = temp;
	}
}