#include "VectorAddition.h"

__global__ void VectorAddition(float* g_A, float* g_B, float* g_C, int Size);
{
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < Size)
	{
		g_C[idx] = g_A[idx] + g_B[idx];
 	}
}

__host__ void GPUAdditionHelper(float* h_A, float* h_B, float* h_C_GPU, const int nSize);
{
	float* dev_A{}, * dev_B{}, * dev_C{}; // pointer pointing to address on GPU memory
	chrono::time_point<std::chrono::system_clock>start, end;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_A, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess)
	{
		cout << "dev_a: cudsMalloc Falied" << endl;
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_B, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess)
	{
		cout << "dev_b: cudsMalloc Falied" << endl;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_C, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess)
	{
		cout << "dev_c: cudsMalloc Falied" << endl;
		goto Error;
	}

	//Copy data from the host to the device global memory
	cudaStatus = cudaMemcpy(dev_A, h_A, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "h_A to dev_A: cudsMemcpy Falied" << endl;
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_B, h_B, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "h_B to dev_B: cudsMemcpy Falied" << endl;
		goto Error;
	}

	//Prepare the thread and block configuration
	int thread_per_block = 1024; 
	int blocks_per_grid = (int)ceil(SIZE / thread_per_block);
	cout << "Vector Size" << SIZE << endl;
	cout << "Vector size in memory (Bytes): " << (SIZE * sizeof(float) / 1e6) << endl;
	cout << "Threads/Block: " << thread_per_block << endl;
	cout << "Block/Grid" << blocks_per_grid << endl;

	//Launch the kernel
	start = std::chrono::system_clock::now();
	VectorAddition << <blocks_per_grid, thread_per_block >> > (dev_A, dev_B, dev_C, nSize);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cout << "Vector addition falied with Error code: " << cudaGetErrorString(cudaStatus) << endl;
		goto Error;
	}
	//wait for the kernal to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaDeviceSynchronize Error with Error code: " << cudaStatus << endl;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double>elapsed_seconds = end - start;
	cout << "GPU execution time: " << (elapsed_seconds.count() * 1000.0f) << " msecs " << endl;
	cudaStatus = cudaMemcpy(h_C_GPU, dev_C, VECTOR_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "h_C to h_C_GPU: cudaMemcpy Falied" << endl;
		goto Error;
	}

Error:
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
}
