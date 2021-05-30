#include "MatrixMult.h"
#include "GPUErrors.h"

__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	if (row < ny && col < nx)
	{
		for (int k = 0; k < nx; k++)
		{
			fSum += g_A[row * nx + k] * g_B[k * nx + col];
		}
		g_C[row * nx + col] = fSum;
	}
}

#define TILE_WIDTH 16
#define PAD 0

//Shared Memory Version

__global__ void SharedMultVer(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Allocate memory on the shared memory to store elements of A and B of the TILE_WIDTH x TILE_WIDTH size equal to a block
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

	//Compute gloabl row and column indexes
	int col = tx + (blockDim.x * bx);
	int row = ty + (blockDim.y * by);

	float fSum = 0.0f;

	for (int tw_idx = 0; tw_idx < (nx / TILE_WIDTH); tw_idx++) //tw_idx represents the index inside the tile
	{
		//Load global element to the tile on the shared memory
		s_A[ty][tx] = g_A[(row * nx) + (tw_idx * TILE_WIDTH) + tx];
		s_B[ty][tx] = g_B[((tw_idx * TILE_WIDTH) + ty) * nx + col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++)
		{
			fSum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[(row * nx) + col] = fSum;

}

__host__ void gpuMult(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx)
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
	cout << "GPU Memory Transfer time (H to D): " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;

	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 16;
	int dimy = 16;

	dim3 block(dimy, dimx);
	dim3 grid((ny + block.y - 1) / block.y, (nx + block.x - 1) / block.x);

	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	HandleCUDAError(cudaEventRecord(kernel_start));
	NaiveMult << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	end = std::chrono::system_clock::now();
	elasped_seconds = end - start;
	MatrixMultVerification(ref, h_C, ny, nx);
	cout << "GPU Memory Transfer time (D to H): " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "Naive Multiplication: GPU Elapsed Time = " << fElapsedTime << " msecs" << endl;

	//Reset the data
	ZeroMatrix(h_C, ny, nx);
	//Release  the GPU Memory for C
	HandleCUDAError(cudaFree(d_C));
	//Reallocate memory on the GPU
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));

	//Multiplication using Shared Memory
	HandleCUDAError(cudaEventRecord(kernel_start));
	SharedMultVer << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	end = std::chrono::system_clock::now();
	elasped_seconds = end - start;
	MatrixMultVerification(ref, h_C, ny, nx);
	cout << "GPU Memory Transfer time (D to H): " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "Multiplication with Shared Memory: GPU Elapsed Time = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaFree(d_C));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}