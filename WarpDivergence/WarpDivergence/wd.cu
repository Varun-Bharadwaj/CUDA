#include "WarpDivergence.h"

__global__ void VectorInitializeWithWD(float* g_C)
{
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float a, b;
	a = b = 0.0f;
	if (idx % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	g_C[idx] = a + b;
}

__global__ void VectorInitializeAcrossWD(float* g_C)
{
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float a, b;
	a = b = 0.0f;
	if ((idx / warpSize) % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	g_C[idx] = a + b;
}

__global__ void VectorInitializeCompiler(float* g_C)
{
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float a, b;
	a = b = 0.0f;
	bool flag = (idx % 2 ==0);
	if (flag)
	{
		a = 100.0f;
	}
	if (!flag)
	{
		b = 200.0f;
	}
	g_C[idx] = a + b;
}

__host__ void VectorOperations(float* h_C)
{
	float* d_C;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;

	float fElapsedTime;

	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_stop);

	//Allocate device memory on the global memory
	cudaMalloc((void**)&d_C, VectorSizeInBytes);


	//Kernel with Warp Divergence
	cudaEventRecord(kernel_start);
	//Launch the Kernel with Warp Divergence
	VectorInitializeWithWD << <1, 32 >> > (d_C);
	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "Kernel with Warp Divergence launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop);
	cudaMemcpy(h_C, d_C, VectorSizeInBytes, cudaMemcpyDeviceToHost);

	cout << "Kernel with Warp Divergence = " << fElapsedTime << " msecs" << endl;

	//Kernel with Across Warp Divergence
	cudaEventRecord(kernel_start);
	//Launch the Kernel with Across Warp Divergence
	VectorInitializeAcrossWD << <1, 32 >> > (d_C);
	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "Kernel a Across Warp Divergence launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop);
	cudaMemcpy(h_C, d_C, VectorSizeInBytes, cudaMemcpyDeviceToHost);

	cout << "Kernel with Across Warp Divergence = " << fElapsedTime << " msecs" << endl;

	//Kernel with Compiler implementation
	cudaEventRecord(kernel_start);
	//Launch the Kernel with Compiler Implementation
	VectorInitializeCompiler << <1, 32 >> > (d_C);
	cudaEventRecord(kernel_stop);
	cudaEventSynchronize(kernel_stop);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "Kernel with Compiler Implementation launch failed with error code: " << cudaGetErrorString(cudaStatus) << endl;
		return;
	}
	cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop);
	cudaMemcpy(h_C, d_C, VectorSizeInBytes, cudaMemcpyDeviceToHost);

	cout << "Kernel with Compiler Implementation = " << fElapsedTime << " msecs" << endl;




	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);
	cudaFree(d_C);
	cudaDeviceReset();
}