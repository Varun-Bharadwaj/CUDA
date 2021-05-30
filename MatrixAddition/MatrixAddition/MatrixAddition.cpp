#include "MatrixAddition.h"

int main()
{
	srand((unsigned)time(NULL));
	int rows = 1 << 8;
	int cols = 1 << 8;
	cout << "Matrix Addition of Size: " << rows << "x" << cols << endl;

	float* A = new float[rows * cols];
	float* B = new float[rows * cols];
	float* C = new float[rows * cols];

	InitializeMatrix(A, rows, cols);
	InitializeMatrix(B, rows, cols);


	MatrixAdditionOnHost(A, B, C, rows, cols);

	DisplayMatrix("A", A, rows, cols);
	DisplayMatrix("B", B, rows, cols);
	DisplayMatrix("C", C, rows, cols);

	float* gpuC = new float[rows * cols];
	cout << "2D Grid and 2D Block Arrangement" << endl;
	MatrixAdditionOnGPU2DG2DB(A, B, gpuC, C, rows, cols);

	cout << endl << "2D Grid and 1D Block Arrangement" << endl;
	MatrixAdditionOnGPU2DG1DB(A, B, gpuC, C, rows, cols);

	cout << endl << "1D Grid and 1D Block Arrangement" << endl;
	MatrixAdditionOnGPU1DG1DB(A, B, gpuC, C, rows, cols);

	delete [] A;
	delete [] B;
	delete [] C;
	delete[] gpuC;
	return 0;
}