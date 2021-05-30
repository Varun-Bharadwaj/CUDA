#include "NLSeries.h"
#include "GPUErrors.h"

//Kernel function to compute the ln(x). 
//Each thread should compute the ln(x) of one value of x.
__global__ void OnComputeNL(float* g_xTemp, float* g_NLTemp, int nSize, int precFactor)
{
	//Determine the scalad thread index
	int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	float fValue, diff;
	int sign = 1;
	if (ix < nSize)
	{
		g_NLTemp[ix] = 0.0;
		diff = g_xTemp[ix] - 1.0; //(x - 1)
		fValue = diff;
		//Implement the code to compute the ln(x) and store in g_NLTemp
		for (int n = 1; n <= precFactor; n++)
		{
			//sign = (n & 1) ? 1 : -1;
			//for (int p = 1; p < n; p++) //This approach is slower, but this doesn't cause overflow
			//{
			//	fValue *= diff;
			//}
			sign = (n & 1) ? 1 : -1;
			int p = n-1;
			while (p)
			{
				if (p & 1) //if p is odd
				{
					fValue *= diff;
				}
				p = p >> 1; // divide p by 2

				diff = diff * diff;
			}
			g_NLTemp[ix] += sign * fValue / (float)n; // claculating ln(x)
		}
	}
	//printf("GPU %f ", g_NLTemp[1]);
}

//GPU Host Function
__host__ void OnGPUComputeNL(float* h_xTemp, float* h_NLTemp, int nSizeTemp, int MTemp)
{
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;
	float fElapsedTime;
	const int SizeInBytes = nSizeTemp * sizeof(float);

	//Instantiate the Event Objects
	HandleCUDAError(cudaEventCreate(&kernel_start));
	HandleCUDAError(cudaEventCreate(&kernel_stop));

	float* d_x;
	float* d_NL;

	//Allocate memory on the GPU for x and ln(x) 
	HandleCUDAError(cudaMalloc((void**)&d_x, SizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_NL,SizeInBytes));

	//Copy x from the host to GPU
	HandleCUDAError(cudaMemcpy(d_x, h_xTemp, SizeInBytes, cudaMemcpyHostToDevice));
	
	//Set up the number of threads per block and number of blocks
	int dimx = 128;
	dim3 block(dimx);
	dim3 grid((nSizeTemp + block.x - 1) / block.x);

	//Set up recording the Event kernel_start
	HandleCUDAError(cudaEventRecord(kernel_start));
	//Launch the kernel OnComputeNL
	OnComputeNL << <grid, block >> > (d_x, d_NL, nSizeTemp, MTemp);
	//Set up recording the Event kernel_stop
	HandleCUDAError(cudaEventRecord(kernel_stop));
	HandleCUDAError(cudaEventSynchronize(kernel_stop));
	//Wait for the the Event kernel_stop to be recorded
	

	GetCUDARunTimeError();
	//Determine the Elapsed Time and store in the fElapsedTime
	HandleCUDAError(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));
	cout << "GPU Execution Time: " << fElapsedTime << " msecs" << endl;

	//Copy d-NL from the GPU to host
	HandleCUDAError(cudaMemcpy(h_NLTemp, d_NL, SizeInBytes, cudaMemcpyDeviceToHost));

	HandleCUDAError(cudaFree(d_x));
	HandleCUDAError(cudaFree(d_NL));
	HandleCUDAError(cudaEventDestroy(kernel_start));
	HandleCUDAError(cudaEventDestroy(kernel_stop));
	HandleCUDAError(cudaDeviceReset());
}