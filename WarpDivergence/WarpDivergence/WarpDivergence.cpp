#include "WarpDivergence.h"

int main()
{
	float* C = new float[SIZE];

	VectorOperations(C);

	delete[] C;

	return 0;
}

