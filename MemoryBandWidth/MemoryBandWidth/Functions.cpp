#include "MemoryBW.h"

void InitializeVector(float* vect, const int nSize)
{
	for (int i = 0; i < nSize; i++)
	{
		vect[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}

