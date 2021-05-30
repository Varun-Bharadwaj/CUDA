#include "MemComparison.h"

void InitVector(int* Vector)
{
	for (int i = 0; i < SIZE; i++)
	{
		Vector[i] = ((int)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}