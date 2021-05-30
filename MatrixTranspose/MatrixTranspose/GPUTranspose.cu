#include "MatrixTranspose.h"
#include "GPUErrors.h"

__global__ void NaiveRowTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixTranspose[ix * ny + iy] = g_Matrix[iy * nx + ix];
	}
}

__global__ void NaiveColTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixTranspose[iy * nx + ix] = g_Matrix[ix * ny + iy];
	}
}

__host__ void TransposeOnGPU(float *h_Matrix, float *h_MatrixTranspose, int ny, int nx)
{
	float *d_Matrix;
	float *d_MatrixTranspose;
	float fMatrixTransferTime = 0.0f;
	const int MatrixSizeInBytes = ny * nx * sizeof(float);

	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_Matrix, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_MatrixTranspose, MatrixSizeInBytes));

	//transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaMemcpy(d_Matrix, h_Matrix, MatrixSizeInBytes, cudaMemcpyHostToDevice));


	//Block and Grid Parameters
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Matrix Transpose Load by Row (Coalesced Access) and Store by column (Stride Access)
	HandleCUDAError(cudaEventRecord(kernel_start));
	NaiveRowTranspose << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	cout << "Load by Row & Store by Column: Matrix Transpose Elapsed Time = " << fElapsedTime << " msecs" << endl;

	//Matrix Transpose Load by Column (Stride Access) and Store by Row (Coalesced Access)
	HandleCUDAError(cudaEventRecord(kernel_start));
	NaiveColTranspose << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	GetCUDARunTimeError();
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));

	cout << "Load by Column & Store by Row: Matrix Transpose Elapsed Time = " << fElapsedTime << " msecs" << endl;

	HandleCUDAError(cudaFree(d_Matrix));
	HandleCUDAError(cudaFree(d_MatrixTranspose));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}

