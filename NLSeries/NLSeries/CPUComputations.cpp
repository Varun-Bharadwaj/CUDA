#include "NLSeries.h"

//function to compute ln(x) on the CPU Naively
void OnCPUNaiveComputeNL(float* xTemp, float* NLTemp, int nSize, int precFactor)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < nSize; i++)
	{
		NLTemp[i] = 0.0;
		for (int n = 1; n <= precFactor; n++)
		{
			NLTemp[i] += (pow(-1.0, (n + 1)) * pow((xTemp[i] - 1.0), n)) / (float)n;
		}
		//cout << NLTemp[i] << endl;
	}
	//cout << NLTemp[1] << endl;
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "Naive CPU Execution Time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}

//function to compute ln(x) on the CPU Optimized
void OnCPUOptimizedComputeNL(float* xTemp, float* NLTemp, int nSize, int precFactor)
{
	float diff, prod;
	int sign = 1;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < nSize; i++)
	{
		NLTemp[i] = 0.0;
		diff = xTemp[i] - 1.0; //(x - 1)
		prod = diff;
		//Implement code to compute ln(x) without using the pow function.
		//Instead of pow function think of doing the same with only a single multiplication in a loop
		for (int n = 1; n <= precFactor; n++)
		{
			sign = (n & 1) ? 1 : -1;
			int p = n-1;
			while (p)
			{
				if (p & 1) //if p is odd
				{
					prod *= diff;
				}
				p = p >> 1; //divide p by 2

				diff = diff * diff;
			}
			NLTemp[i] += sign * prod / (float)n;//calculating ln(x)
		}
		//cout << NLTemp[i] << endl;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "Optimized CPU Execution Time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}
