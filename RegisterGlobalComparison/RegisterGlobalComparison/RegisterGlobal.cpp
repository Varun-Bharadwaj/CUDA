#include "MemComparison.h"

int main()
{
	int* Vector = new int[SIZE];
	int* Result = new int[ITERATION_COUNT];
	InitVector(Vector);
	MemoryBenchmark(Vector, Result);
	delete[] Vector;
	delete[] Result;
	return 0;
}