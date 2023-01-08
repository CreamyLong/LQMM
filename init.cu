#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
//#include <sys/time.h> //only windows
#include <time.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include "pbc.h"

// randome number generator [0.0-1.0)
float R2S()
{
	int ran = rand();
	float fran = (float)ran / (float)RAND_MAX;
	return fran;
}

void init(
	unsigned int np,
	float4* r,
	float4* v,
	float3 box,
	float min_dis)
{
	for (unsigned int i = 0; i < np; i++)
	{
		bool find_pos = false;
		float4 ri;
		while (!find_pos)
		{
			ri.x = (R2S() - 0.5) * box.x;
			ri.y = (R2S() - 0.5) * box.y;
			ri.z = (R2S() - 0.5) * box.z;
			find_pos = true;

			for (unsigned int j = 0; j < i; j++)
			{
				float dx = pbc(ri.x - r[j].x, box.x);
				float dy = pbc(ri.y - r[j].y, box.y);
				float dz = pbc(ri.z - r[j].z, box.z);
				float r = sqrt(dx * dx + dy * dy + dz * dz);
				if (r < min_dis)                           // a minimum safe distance to avoid the overlap of LJ particles
				{
					find_pos = false;
					break;
				}
			}
		}
		if (R2S() > 0.5) // randomly generate the type of particle, 1.0 represent type A and 2.0 represent type B 
			ri.w = 1.0;
		else
			ri.w = 2.0;

		r[i] = ri;
		v[i].w = 1.0;
	}
}