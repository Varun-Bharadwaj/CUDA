#include "DataLayout.h"


int main()
{
	srand(time(NULL));
	cout << "Number of Elements of the Array of Structures (AoS) = " << LEN << endl;
	cout << "\tSize of the Array of Structures (AoS) in Bytes = " << LEN * sizeof(AoS) << endl;
	AoS* InputAoS = new AoS[LEN];
	AoS* OutputAoS = new AoS[LEN];
	InitializeAoS(InputAoS, LEN);
	cpuComputeAoS(InputAoS, OutputAoS, LEN);
	/*Write code to display the address of the first four data structure variables 
	in the InputAoS of AoS Type*/

	cout << "\n\nNumber of Elements of the Structure of Arrays (SoA) = " << LEN << endl;
	cout << "\tSize of the Structures of Arrays (SoA) in Bytes = " << sizeof(SoA) << endl;
	SoA* InputSoA = new SoA;
	SoA* OutputSoA = new SoA;
	Copy_AoSToSoA(InputAoS, InputSoA, LEN);
	cpuComputeSoA(InputSoA, OutputSoA, LEN);
	/*Write code to display the address of the first four address of array x and array y 
	in the structure InputSoA of SoA Type*/

	//Verification on CPU
	VerifyOnCPU(OutputAoS, OutputSoA, LEN);

	//GPU Array of Structures Implementation
	AoS* gpuOutputAoS = new AoS[LEN];
	GPUAoS(InputAoS, gpuOutputAoS, LEN);
	VerifyGPUAoS(OutputAoS, gpuOutputAoS, LEN);

	//GPU Structure of Arrays Implementation
	SoA* gpuOutputSoA = new SoA;
	GPUSoA(InputSoA, gpuOutputSoA, LEN);
	VerifyGPUSoA(OutputSoA, gpuOutputSoA, LEN);


	delete[] gpuOutputAoS;
	delete gpuOutputSoA;

	delete[] InputAoS;
	delete[] OutputAoS;

	delete InputSoA;
	delete OutputSoA;
	return 0;
}