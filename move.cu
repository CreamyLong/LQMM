#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include "pbc.h"
#include "utils.h"


// first step integration of velocity verlet algorithm
extern "C" __global__ void first_integration_kernel(
	unsigned int np,
	float dt,
	float3 box,
	float4 * r,
	float4 * v,
	float4 * f)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < np)
	{
		float4 ri = r[i];
		float mass = v[i].w;
		
		v[i].x += 0.5 * dt * f[i].x / mass;
		v[i].y += 0.5 * dt * f[i].y / mass;
		v[i].z += 0.5 * dt * f[i].z / mass;

		ri.x += dt * v[i].x;
		ri.y += dt * v[i].y;
		ri.z += dt * v[i].z;

		r[i].x = pbc(ri.x, box.x);
		r[i].y = pbc(ri.y, box.y);
		r[i].z = pbc(ri.z, box.z);
	}
}

void first_integration(
	unsigned int np,
	float dt,
	float3 box,
	float4* r,
	float4* v,
	float4* f,
	unsigned int block_size)
{

	dim3 grid((np / block_size) + 1, 1, 1);
	dim3 block(block_size, 1, 1);

	first_integration_kernel << < grid, block >> > (np, dt, box, r, v, f);
	
	// block until the device has completed
	cudaDeviceSynchronize(); 	


	// check if kernel execution generated an error
	checkCUDAError("kernel execution");
}

// second step integration of velocity verlet algorithm
extern "C" __global__ void second_integration_kernel(
	unsigned int np,
	float dt,
	float4 * v,
	float4 * f)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < np)
	{
		float mass = v[i].w;
		v[i].x += 0.5 * dt * f[i].x / mass;
		v[i].y += 0.5 * dt * f[i].y / mass;
		v[i].z += 0.5 * dt * f[i].z / mass;
	}
}

void second_integration(
	unsigned int np,
	float dt,
	float4* v,
	float4* f,
	unsigned int block_size)
{

	dim3 grid((np / block_size) + 1, 1, 1);
	dim3 block(block_size, 1, 1);

	second_integration_kernel << < grid, block >> > (np, dt, v, f);
	// block until the device has completed

	cudaDeviceSynchronize();

	// check if kernel execution generated an error
	checkCUDAError("kernel execution");
}