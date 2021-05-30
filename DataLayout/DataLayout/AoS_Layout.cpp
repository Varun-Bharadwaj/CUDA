#include "DataLayout.h"

void InitializeAoS(AoS* aos, int size)
{
	for (int i = 0; i < size; i++)
	{
		aos[i].x = (float)(rand() & 0xFF) / 100.0f;
		aos[i].y = (float)(rand() & 0xFF) / 100.0f;
	}
}

void cpuComputeAoS(AoS* in, AoS* out, int size)
{
	AoS temp;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < size; i++)
	{
		temp = in[i];
		temp.x += 10.0f;
		temp.y += 20.0f;
		out[i] = temp;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time for Array of Structures (AoS) : " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "Address of AOS x " << &in[0].x << endl;
	cout << "Address of AOS x " << &in[1].x << endl;
	cout << "Address of AOS x " << &in[2].x << endl;
	cout << "Address of AOS x " << &in[3].x << endl;
	cout << "" << endl;
	cout << "Address of AOS y " << &in[0].y << endl;
	cout << "Address of AOS y " << &in[1].y << endl;
	cout << "Address of AOS y " << &in[2].y << endl;
	cout << "Address of AOS y " << &in[3].y << endl;
}