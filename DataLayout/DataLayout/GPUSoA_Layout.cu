#include "DataLayout.h"
#include "GPUErrors.h"

__host__ void GPUSoA(SoA* h_in, SoA* h_out, int size)
{
	SoA* d_in, * d_out;
	const int SizeInBytes_SoA = sizeof(SoA);
	cout << "\n\nGPU Execution using Structure of Arrays (SoA)" << endl;
	cout << "\tStructure of Arrays (SoA) Size in Bytes = " << SizeInBytes_SoA << " bytes" << endl;

	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((SoA**)&d_in, SizeInBytes_SoA));
	HandleCUDAError(cudaMalloc((AoS**)&d_out, SizeInBytes_SoA));

	//Transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaMemcpy(d_in, h_in, SizeInBytes_SoA, cudaMemcpyHostToDevice));

	//Kernel Invoke Parameters - 1D Grid and 1D Blocks
	int dimx = 256;
	dim3 block(dimx, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);

	cout << "\t1D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\t1D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;

	HandleCUDAError(cudaEventRecord(kernel_start));
	gpuComputeSoA << <grid, block >> > (d_in, d_out, size);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	HandleCUDAError(cudaMemcpy(h_out, d_out, SizeInBytes_SoA, cudaMemcpyDeviceToHost));
	cout << "\tGPU Computation using Structure of Arrays (SoA): Elapsed Time = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_in));
	HandleCUDAError(cudaFree(d_out));

	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}

__global__ void gpuComputeSoA(SoA* g_in, SoA* g_out, int n)
{
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < n)
	{
		float tempX = g_in->x[idx];
		float tempY = g_in->y[idx];

		tempX += 10.0f;
		tempY += 20.0f;

		g_out->x[idx] = tempX;
		g_out->y[idx] = tempY;
	}
}