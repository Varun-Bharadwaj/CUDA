#include "NearestNeighbors.h"

void CPU_FindNearestNeighbors(float3* Points, int* ClosestPointIndex, int PointCount)
{
	if (PointCount <= 1)
	{
		return;
	}

	float NeighborClosest;
	float distance;
	
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	//Loop through every point
	for (int currPoint = 0; currPoint < PointCount; currPoint++)
	{
		//Assume nearest neighbhor is very far
		NeighborClosest = 3.40282E38f;
		//Loop through every point again
		for (int i = 0; i < PointCount; i++)
		{
			//Do not check distance between the same point
			if (i == currPoint)
			{
				continue;
			}

			//Compute the distance between Points[currPoint] and Points[i]
			distance = sqrtf(((Points[currPoint].x - Points[i].x) * (Points[currPoint].x - Points[i].x) + (Points[currPoint].y - Points[i].y) * (Points[currPoint].y - Points[i].y) + (Points[currPoint].z - Points[i].z) * (Points[currPoint].z - Points[i].z)));

			//Is the computed distance nearest
			if (distance < NeighborClosest)
			{
				//Update the nearest neighbor distance
				NeighborClosest = distance;
				//Update the index of the nearest neighbor
				ClosestPointIndex[currPoint] = i;
			}
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}