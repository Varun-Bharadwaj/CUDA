#include "NLSeries.h"

//Function to generate values of x between x_min and x_max based on the SIZE values
void OnGenerateX(float* xTemp)
{
	float Spread = SIZE;
	float fIncrementValue = (x_max - x_min) / Spread;
	xTemp[0] = x_min;
	for (int i = 1; i < SIZE; i++)
	{
		xTemp[i] = x_min + i * fIncrementValue;
	}
}

//Computed ln(x) using the library function to be used as the reference
void OnGenerateRef(float* xTemp, float* h_NLTemp, int nSizeTemp)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < nSizeTemp; i++)
	{
		h_NLTemp[i] = logf(xTemp[i]);
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "Reference CPU Execution Time: " << (elasped_seconds.count() * 1000.0f) << " msecs"<<endl;
}

//Function displays the ln(x) of the first and last five values of x.
void OnPrintResults(float* xTemp, float* RefNL, float* CPUNaiveNL, float* CPUOptNL, float* GPUNL, int nSize)
{
	//Code to verify results by inspection
	cout << "x\t\tRef\t\tCPU-Naive\t\tCPU-Opt\t\tGPU" << endl;
	for (int i = 0; i < 5; i++)
	{
		cout << setprecision(6)<<xTemp[i] << "\t\t" << RefNL[i] << "\t\t" << CPUNaiveNL[i] << "\t\t" << CPUOptNL[i] << "\t\t" << GPUNL[i] << endl;
	}
	for (int i = nSize - 5; i < nSize; i++)
	{
		cout << setprecision(6)<<xTemp[i] << "\t\t" << RefNL[i] << "\t\t" << CPUNaiveNL[i] << "\t\t" << CPUOptNL[i] << "\t\t" << GPUNL[i] << endl;
	}
}