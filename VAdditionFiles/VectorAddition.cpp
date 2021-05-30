#include "VectorAddition.h"

int main()
{
	float* h_A, * h_B, * h_C_CPU, * h_C_GPU;
	chrono::time_point<std::chrono::system_clock> start, end;

	//Memory Allocation
	h_A = new float[SIZE];
	h_B = new float[SIZE];
	h_C_CPU = new float[SIZE];
	h_C_GPU = new float[SIZE];

	//Initialize Data
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);

	start = std::chrono::system_clock::now();
	CPUVectorAddition(h_A, h_B, h_C_CPU, SIZE);
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;

	DisplayVector("A", h_A, SIZE);
	DisplayVector("B", h_B, SIZE);
	DisplayVector("C", h_C_CPU, SIZE);
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	//Perform Vector Addition on GPU
	


	delete[] h_A;
	delete[] h_B;
	delete[] h_C_CPU;
	delete[] h_C_GPU;

	return 0;
}