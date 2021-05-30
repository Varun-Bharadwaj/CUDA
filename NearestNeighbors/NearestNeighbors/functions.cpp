#include "NearestNeighbors.h"

void GenerateRandom3DPoints(float3* random3DPoints, int PointCount)
{
	for (int i = 0; i < PointCount; i++)
	{
		random3DPoints[i].x = (float)((rand() % 10000) - 5000);
		random3DPoints[i].y = (float)((rand() % 10000) - 5000);
		random3DPoints[i].z = (float)((rand() % 10000) - 5000);
	}
}

void VerifyResults(int* CPU_ClosestPointIndex, int* GPU_ClosestPointIndex, int PointCount)
{
	for (int i = 0; i < PointCount; i++)
	{
		if (CPU_ClosestPointIndex[i] != GPU_ClosestPointIndex[i])
		{
			cout << "Error at Point - " << i << endl;
			return;
		}
	}
	cout << "\tSuccessful Verification with CPU Computations" << endl;
}

void PrintNearestPoint(int* ClosestPointIndex, int PointCount)
{
	if (PointCount < 10)
	{
		return;
	}
	cout << "Closest Points of the First Ten Points" << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << "Point[" << i << "] <-> " << "Point[" << ClosestPointIndex[i] << "]" << endl;
	}
}