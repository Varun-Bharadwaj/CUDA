#include "NearestNeighbors.h"

int main()
{
	float3* Points3D = new float3[NUMBER_OF_POINTS];
	int* NearestNeighborPointIndexByCPU = new int[NUMBER_OF_POINTS];
	int* NearestNeighborPointIndexByGPU = new int[NUMBER_OF_POINTS];
	srand((unsigned)time(NULL));

	//Generate Random Points
	GenerateRandom3DPoints(Points3D, NUMBER_OF_POINTS);
	//Initialize the NearestNeighborPointIndex vectors with zero
	memset(NearestNeighborPointIndexByCPU, 0, (NUMBER_OF_POINTS * sizeof(int)));
	memset(NearestNeighborPointIndexByGPU, 0, (NUMBER_OF_POINTS * sizeof(int)));
	cout << "Number of 3D Points to compute nearest neighbors: " << NUMBER_OF_POINTS << endl;
	//CPU Computation
	CPU_FindNearestNeighbors(Points3D, NearestNeighborPointIndexByCPU, NUMBER_OF_POINTS);
	PrintNearestPoint(NearestNeighborPointIndexByCPU, NUMBER_OF_POINTS);

	cout << "GPU Implementation" << endl;
	//GPU Implementation using only Global Memory
	GPUGlobalNearestNeighbors(Points3D, NearestNeighborPointIndexByGPU, NUMBER_OF_POINTS);
	VerifyResults(NearestNeighborPointIndexByCPU, NearestNeighborPointIndexByGPU, NUMBER_OF_POINTS);

	////Uncomment if you want to attempt the extra credit problem
	////Reinitialize the GPU Index
	//memset(NearestNeighborPointIndexByGPU, 0, (NUMBER_OF_POINTS * sizeof(int)));
	////GPU Implementation using Global and Shared Memory
	//GPUSharedNearestNeighbors(Points3D, NearestNeighborPointIndexByGPU, NUMBER_OF_POINTS);
	//VerifyResults(NearestNeighborPointIndexByCPU, NearestNeighborPointIndexByGPU, NUMBER_OF_POINTS);

	delete[] NearestNeighborPointIndexByGPU;
	delete[] NearestNeighborPointIndexByCPU;
	delete[] Points3D;
	return 0;
}