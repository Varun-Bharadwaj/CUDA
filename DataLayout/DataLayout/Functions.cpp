#include "DataLayout.h"

void VerifyOnCPU(AoS* aos, SoA* soa, int size)
{
	float fTolerance = 1.0E-04;
	for (int i = 0; i < size; i++)
	{
		if (fabs(aos[i].x - soa->x[i]) > fTolerance)
		{
			cout << "\tError on CPU" << endl;
			cout << "\tAoS[" << (i + 1) << "].x = " << aos[i].x << endl;
			cout << "\tSoA->x[" << (i + 1) << "] = " << soa->x[i] << endl;
			return;
		}
		if (fabs(aos[i].y - soa->y[i]) > fTolerance)
		{
			cout << "\tError on CPU" << endl;
			cout << "\tAoS[" << (i + 1) << "].y = " << aos[i].y << endl;
			cout << "\tSoA->y[" << (i + 1) << "] = " << soa->y[i] << endl;
			return;
		}
	}
}

void VerifyGPUAoS(AoS* cpu_aos, AoS* gpu_aos, int size)
{
	float fTolerance = 1.0E-04;
	for (int i = 0; i < size; i++)
	{
		if (fabs(cpu_aos[i].x - gpu_aos[i].x) > fTolerance)
		{
			cout << "\tError on GPU-AoS" << endl;
			cout << "\tAoS[" << (i + 1) << "].x = " << cpu_aos[i].x << endl;
			cout << "\tAoS[" << (i + 1) << "].x = " << gpu_aos[i].x << endl;
			return;
		}
		if (fabs(cpu_aos[i].y - gpu_aos[i].y) > fTolerance)
		{
			cout << "\tError on GPU-AoS" << endl;
			cout << "\tAoS[" << (i + 1) << "].y = " << cpu_aos[i].y << endl;
			cout << "\tAoS[" << (i + 1) << "].y = " << gpu_aos[i].y << endl;
			return;
		}
	}
}

void VerifyGPUSoA(SoA* cpu_soa, SoA* gpu_soa, int size)
{
	float fTolerance = 1.0E-04;
	for (int i = 0; i < size; i++)
	{
		if (fabs(cpu_soa->x[i] - gpu_soa->x[i]) > fTolerance)
		{
			cout << "\tError on GPU-SoA" << endl;
			cout << "\tSoA->x[" << (i + 1) << "] = " << cpu_soa->x[i] << endl;
			cout << "\tSoA->x[" << (i + 1) << "] = " << gpu_soa->x[i] << endl;
			return;
		}
		if (fabs(cpu_soa->y[i] - gpu_soa->y[i]) > fTolerance)
		{
			cout << "\tError on GPU-SoA" << endl;
			cout << "\tSoA->y[" << (i + 1) << "] = " << cpu_soa->y[i] << endl;
			cout << "\tSoA->y[" << (i + 1) << "] = " << gpu_soa->y[i] << endl;
			return;
		}
	}
}