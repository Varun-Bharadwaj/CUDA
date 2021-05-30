#include "NLSeries.h"

int main()
{
	float* fX; //Pointer to the vector containing value of x
	float* fRefNL; //Pointer to the vector to store the ln(x) values computed using the math library function.
	float* fCpuNL; //Pointer to the vector used to store the ln(x) values computed using CPU without optimization
	float* fCpuOptNL; //Pointer to the vector used to store the ln(x) values computed using CPU with optimization
	float* fGpuNL; //Pointer to the vector used to store the ln(x) values computed using GPU

	fX = new float[SIZE] {};
	fRefNL = new float[SIZE] {};
	fCpuNL = new float[SIZE] {};
	fCpuOptNL = new float[SIZE] {};
	fGpuNL = new float[SIZE] {};

	//Call the function OnGenerateX to populate the vector fX with values between 0.1 and 2.0
	OnGenerateX(fX);
	//Call the function OnGenerateRef to compute the ln(x) using the built-in math library function
	OnGenerateRef(fX, fRefNL, SIZE);

	//Call the function OnCPUNaiveComputeNL to compute the ln(x) using the naive CPU Implementation
	OnCPUNaiveComputeNL(fX, fCpuNL, SIZE, M);

	//Call the function OnCPUOptimizedComputeNL to compute the ln(x) using the optimized CPU Implementation
	OnCPUOptimizedComputeNL(fX, fCpuOptNL, SIZE, M);

	//Call the function OnGPUComputeNL to compute the ln(x) using the GPU Implementation
	OnGPUComputeNL(fX, fGpuNL, SIZE, M);

	//Display the results
	OnPrintResults(fX, fRefNL, fCpuNL, fCpuOptNL, fGpuNL, SIZE);

	delete [] fX;
	delete [] fRefNL;
	delete [] fCpuNL;
	delete [] fCpuOptNL;
	delete [] fGpuNL;
	return 0;
}