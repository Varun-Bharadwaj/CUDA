#include "MatrixAddition.h"

void MatrixAdditionOnHost(float* A, float* B, float* C, const int ny, const int nx)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	float* p = A;
	float* q = B;
	float* r = C;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			r[j] = p[j] + q[j];
		}
		p += nx;
		q += nx;
		r += nx;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}