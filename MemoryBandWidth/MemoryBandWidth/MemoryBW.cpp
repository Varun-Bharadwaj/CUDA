#include "MemoryBW.h"

int main()
{
	float* h_x, * h_y;
	chrono::time_point<std::chrono::system_clock> start, end;

	int N = 20 * (1 << 20);

	//Memory Allocation
	h_x = new float[N];
	h_y = new float[N];

	//Initialize Data
	InitializeVector(h_x, N);
	InitializeVector(h_y, N);

	float a = 0.1;
	MemoryBenchMarks(a,h_x, h_y, N);

	delete[] h_x;
	delete[] h_y;
	return 0;
}