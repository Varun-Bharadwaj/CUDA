#include "DataLayout.h"

void Copy_AoSToSoA(AoS* aos, SoA* soa, int size)
{
	for (int i = 0; i < size; i++)
	{
		soa->x[i] = aos[i].x;
		soa->y[i] = aos[i].y;
	}
}

void cpuComputeSoA(SoA* in, SoA* out, int size)
{
	float tempX, tempY;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < size; i++)
	{
		tempX = in->x[i];
		tempY = in->y[i];

		tempX += 10.0f;
		tempY += 20.0f;

		out->x[i] = tempX;
		out->y[i] = tempY;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time for Structure of Array (SoA) : " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "Address of SOA x " << &in->x[0] << endl;
	cout << "Address of SOA x " << &in->x[1] << endl;
	cout << "Address of SOA x " << &in->x[2] << endl;
	cout << "Address of SOA x " << &in->x[3] << endl;
	cout << "" << endl;
	cout << "Address of SOA y " << &in->y[0] << endl;
	cout << "Address of SOA y " << &in->y[1] << endl;
	cout << "Address of SOA y " << &in->y[2] << endl;
	cout << "Address of SOA y " << &in->y[3] << endl;
}