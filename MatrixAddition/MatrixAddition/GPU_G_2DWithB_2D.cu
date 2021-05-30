#include "MatrixAddition.h"
#include "GPUErrors.h"

//2D Kernel
__global__ void MatrixAddition2DG2DB(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int ix = threadIdx.x + (blockIdx.x * blockDim.x); //Index of thread along x direction
	int iy = threadIdx.y + (blockIdx.y * blockDim.y);
	//Linear or Scalar index of thread
	int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
	{
		g_C[idx] = g_A[idx] + g_B[idx];
	}
}


//GPU Host Function
__host__ void MatrixAdditionOnGPU2DG2DB(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx)
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

	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx,dimy); //creates a 2d block

	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Block along X dimension: " << grid.x << endl;
	cout << "\tNumber of Block along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	HandleCUDAError(cudaEventRecord(kernel_start));
	MatrixAddition2DG2DB << <grid, block >> > (d_A, d_B, d_C, nx, ny);
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
	cout << "2DG2DB Elapsed Time (GPU) = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaFree(d_C));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}