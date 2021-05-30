#include "NearestNeighbors.h"
#include "GPUErrors.h"

__global__ void gpuGlobalNN(float3* g_Points, int* g_ClosestPointIndex, int count)
{
	int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	//int tid = threadIdx.x;
	float NeighborClosest;
	float distance;
	//float3* blockad = g_Points + (blockIdx.x * blockDim.x * 2);

	if (ix < count)
	{
		//Assume nearest neighbhor is very far
		NeighborClosest = 3.40282E38f;
		//Loop through every point again
		for (int i = 0; i < count; i++)
		{
			//Do not check distance between the same point
			if (i ^ ix)
			{
				//Compute the distance between Points[ix] and Points[i]
				distance = sqrtf(((g_Points[ix].x - g_Points[i].x) * (g_Points[ix].x - g_Points[i].x) + (g_Points[ix].y - g_Points[i].y) * (g_Points[ix].y - g_Points[i].y) + (g_Points[ix].z - g_Points[i].z) * (g_Points[ix].z - g_Points[i].z)));
				//Is the computed distance nearest
				if (distance < NeighborClosest)
				{
					//Update the nearest neighbor distance
					NeighborClosest = distance;
					//Update the index of the nearest neighbor
					g_ClosestPointIndex[ix] = i;
				}
			}


		}

	}
}

//Host Function
__host__ void GPUGlobalNearestNeighbors(float3* Points, int* ClosestPointIndex, int PointCount)
{
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;


	float fElapsedTime = 0.0f;
	float fMemoryTransferTime = 0.0f;
	const int SizeInBytes = PointCount * sizeof(float3);
	const int SizeInBytes1 = PointCount * sizeof(int);

	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	float3* d_Points;
	int* d_ClosestPointIndex;

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_Points, SizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_ClosestPointIndex, SizeInBytes1));

	/*transfer data from CPU Memory to GPU Memory,measure the memory copy time, and 
	store in the fMemoryTransferTime*/
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(d_Points, Points, SizeInBytes, cudaMemcpyHostToDevice));
	end = std::chrono::system_clock::now();
	//Store the memory copy time in the variable
	std::chrono::duration<float> elapsed_seconds = end - start;

	fMemoryTransferTime = elapsed_seconds.count() * 1000.0f;
	//cout << "\nGPU Computation using only Global Memory" << endl;
	//Develop Block and Grid Parameters and display
	int dimx = 256;
	dim3 block(dimx);
	dim3 grid(PointCount + block.x - 1 / block.x);

	//Computations using only the global memory
	//Launch the start Event Timer
	HandleCUDAError(cudaEventRecord(kernel_start));
	//Launch the kernel
	gpuGlobalNN << <grid, block >> > (d_Points, d_ClosestPointIndex, PointCount);
	//Launch the stop Event Timer
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	//Block the CPU for the stop event to occur
	
	GetCUDARunTimeError();
	//Compute the Elapsed Time
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));
	/*transfer data from GPU Memory to CPU Memory, measure the memory copy time, and 
	update the fMemoryTransferTime*/
	start = std::chrono::system_clock::now();
	HandleCUDAError(cudaMemcpy(ClosestPointIndex, d_ClosestPointIndex, SizeInBytes1, cudaMemcpyDeviceToHost));
	end = std::chrono::system_clock::now();
	std::chrono::duration<float> elapsed_seconds1 = end - start;

	fMemoryTransferTime += elapsed_seconds1.count() * 1000.0f;
	cout << "\tTotal Memory Transfer Time (H->D and H<-D): " << fMemoryTransferTime << " msecs" << endl;
	cout << "\tNaive GPU Nearest Neighborhood Computation Time using only Global Memory: " << fElapsedTime << " msecs" << endl;

	//Release the memory on the GPU
	HandleCUDAError(cudaFree(d_Points));
	HandleCUDAError(cudaFree(d_ClosestPointIndex));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	//Reset the GPU device
	HandleCUDAError(cudaDeviceReset());
}