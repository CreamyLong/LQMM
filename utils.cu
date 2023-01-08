#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
//#include <sys/time.h> //only windows
#include <time.h>

#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include "device_functions.h"
#include <device_launch_parameters.h>

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

// output system information and frame in XYZ formation which can be read by VMD
void output(
	FILE* traj,
	unsigned int step,
	float* info,
	float4* r,
	unsigned int np)
{
	float temp = info[0];
	float potential = info[1];
	float energy = info[2];

	fprintf(traj, "%d\n  step=%d  temp=%20.8f  pot=%20.8f  ener=%20.8f\n", np, step, temp, potential, energy);
	for (unsigned int i = 0; i < np; i++)
	{
		float4 ri = r[i];
		if (ri.w == 1.0)
			fprintf(traj, "A  %20.8f %20.8f %20.8f\n", ri.x, ri.y, ri.z);
		else if (ri.w == 2.0)
			fprintf(traj, "B  %20.8f %20.8f %20.8f\n", ri.x, ri.y, ri.z);
	}
}