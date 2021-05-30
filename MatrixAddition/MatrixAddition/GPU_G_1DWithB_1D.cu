#include "MatrixAddition.h"
#include "GPUErrors.h"

__global__ void MatrixAddition1DG1DB(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	if (ix < nx)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			int idx = iy * nx + ix;
			g_C[idx] = g_A[idx] + g_B[idx];
		}
	}
}


//GPU Host Function
__host__ void MatrixAdditionOnGPU1DG1DB(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx)
{
	float* d_A, * d_B, * d_C;
	const int MatrixSizeInBytes = ny * nx * sizeof(float);

	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_B, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));

	//transfer data from CPU Memory to GPU Memory
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	HandleCUDAError(cudaMemcpy(d_B, h_B, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "Memory Copy - HostToDevice: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;

	//Kernel Invoke Parameters - 1D Grid and 1D Blocks
	int dimx = 256;
	dim3 block(dimx);

	dim3 grid((nx + block.x - 1) / block.x);

	cout << "1D Grid Dimension" << endl;
	cout << "\tNumber of Block along X dimension: " << grid.x << endl;
	cout << "1D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;

	HandleCUDAError(cudaEventRecord(kernel_start));
	MatrixAddition1DG1DB << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();

	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	end = std::chrono::system_clock::now();
	elasped_seconds = end - start;
	MatrixAdditionVerification(ref, h_C, ny, nx);
	cout << "Memory Copy - DeviceToHost: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "1DG1DB Elapsed Time (GPU) = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaFree(d_C));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}
