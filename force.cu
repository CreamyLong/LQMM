#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include "device_functions.h" //__syncthreads
#include <device_launch_parameters.h>

#include "pbc.h"
#include "utils.h"

#define MIN(A, B) A < B ? A : B;

// non-bonded force calculation	
extern "C" __global__ void force_calculation_kernel(
	unsigned int np,
	float3 box,
	float3 lj1,
	float3 lj2,
	float4 * r,
	float4 * f,
	float rcutsq)
{
	extern __shared__ float4 spos[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float4 force = make_float4(0.0, 0.0, 0.0, 0.0);
	float4 ri = make_float4(0.0, 0.0, 0.0, 0.0);
	
	if (i < np)
		ri = r[i];

	for (int start = 0; start < np; start += blockDim.x)
	{
		// load data
		float4 posj = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if (start + threadIdx.x < np)
			posj = r[start + threadIdx.x];
		__syncthreads();

		spos[threadIdx.x] = posj;
		__syncthreads();

		int end_offset = blockDim.x;
		end_offset = MIN(end_offset, np - start);

		if (i < np)
		{
			for (int cur_offset = 0; cur_offset < end_offset; cur_offset++)
			{
				float4 rj = spos[cur_offset];
				int j = start + cur_offset;

				/* particles have no interactions with themselves */
				if (i == j) continue;

				/* calculated the shortest distance between particle i and j */

				float dx = pbc(ri.x - rj.x, box.x);
				float dy = pbc(ri.y - rj.y, box.y);
				float dz = pbc(ri.z - rj.z, box.z);
				float type = ri.w + rj.w;

				float rsq = dx * dx + dy * dy + dz * dz;

				/* compute force and energy if within cutoff */
				if (rsq < rcutsq)
				{
					float lj1_ij, lj2_ij;
					if (type == 2.0)               // i=1.0, j=1.0
					{
						lj1_ij = lj1.x;
						lj2_ij = lj2.x;
					}
					else if (type == 3.0)         // i=1.0, j=2.0; or i=2.0, j=1.0
					{
						lj1_ij = lj1.y;
						lj2_ij = lj2.y;
					}
					else if (type == 4.0)         // i=2.0, j=2.0
					{
						lj1_ij = lj1.z;
						lj2_ij = lj2.z;
					}

					float r2inv = float(1.0) / rsq;
					float r6inv = r2inv * r2inv * r2inv;

					// force between particle i and j
					float ffac = r2inv * r6inv * (float(12.0) * lj1_ij * r6inv - float(6.0) * lj2_ij);
					
					// potential between particle i and j
					float epot = r6inv * (lj1_ij * r6inv - lj2_ij);

					force.x += ffac * dx;
					force.y += ffac * dy;
					force.z += ffac * dz;
					force.w += epot;
				}
			}
		}
	}

	if (i < np) {
		f[i] = force;
		//printf("%d  %f   %f   %f   %f  \n", i, force.x,  force.y, force.z, force.w);
	}
}


void force_calculation(
	unsigned int np,
	float3 box,
	float3 epsilon,
	float3 sigma,
	float4* r,
	float4* f,
	float rcut,
	unsigned int block_size)
{

	dim3 grid((np / block_size) + 1, 1, 1);
	dim3 block(block_size, 1, 1);
	unsigned int shared_bytes = sizeof(float4) * block_size;
	force_calculation_kernel << < grid, block, shared_bytes >> > (np, box, epsilon, sigma, r, f, rcut);
	// block until the device has completed
	//cudaThreadSynchronize();
	cudaDeviceSynchronize();

	// check if kernel execution generated an error
	checkCUDAError("kernel execution");
}