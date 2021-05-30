#include "MatrixMult.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 1 << 10;
	int cols = 1 << 10;
	cout << "Matrix Multiplication of Size: " << rows << "x" << cols << endl;
	float* A, * B, * C;

	A = new float[rows * cols];
	B = new float[rows * cols];
	C = new float[rows * cols];

	/*HandleCUDAError(cudaHostAlloc((float**)&A, (rows * cols * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((float**)&B, (rows * cols * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((float**)&C, (rows * cols * sizeof(float)), cudaHostAllocDefault));*/
	InitializeMatrix(A, rows, cols);
	InitializeMatrix(B, rows, cols);

	//Host Multiplication
	cpuMatrixMult(A, B, C, rows, cols);

	DisplayMatrix("A", A, rows, cols);
	DisplayMatrix("B", B, rows, cols);
	DisplayMatrix("C", C, rows, cols);

	float* gpuC;
	gpuC = new float[rows * cols];

	//HandleCUDAError(cudaHostAlloc((float**)&gpuC, (rows * cols * sizeof(float)), cudaHostAllocDefault));
	gpuMult(A, B, gpuC, C, rows, cols);

	/*cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(gpuC);*/

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] gpuC;

	return 0;
}