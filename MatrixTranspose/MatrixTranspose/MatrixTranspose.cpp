#include "MatrixTranspose.h"


int main()
{
	srand((unsigned)time(NULL));

	//Setup Matrix
	int cols = 1 << 12;
	int rows = 1 << 12;
	cout << "Matrix Size = " << rows << " x " << cols << endl;

	float *Matrix = new float[rows*cols];
	float *CPUMatrixTranspose = new float[rows*cols];
	float *GPUMatrixTranspose = new float[rows*cols];

	InitializeMatrix(Matrix, rows, cols);
	ZeroMatrix(CPUMatrixTranspose, rows, cols);
	TransposeOnCPU(Matrix, CPUMatrixTranspose, rows, cols);


	//Determining Performance Bounds for Transposing a matrix
	PerformanceBounds(Matrix, rows, cols);

	//Naive Transpose
	ZeroMatrix(GPUMatrixTranspose, rows, cols);
	TransposeOnGPU(Matrix, GPUMatrixTranspose, rows, cols);
	VerifyTranspose(GPUMatrixTranspose, CPUMatrixTranspose, rows, cols);

	delete[] Matrix;
	delete[] CPUMatrixTranspose;
	delete[] GPUMatrixTranspose;
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
