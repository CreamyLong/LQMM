#pragma once
#include <cuda_runtime.h> //float3 ,float4


void force_calculation(
	unsigned int np,
	float3 box,
	float3 epsilon,
	float3 sigma,
	float4* r,
	float4* f,
	float rcut,
	unsigned int block_size);