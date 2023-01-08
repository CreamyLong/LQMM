#pragma once
#include <cuda_runtime.h> //float3 ,float4

void first_integration(unsigned int np, float dt, float3 box, float4* r, float4* v, float4* f, unsigned int block_size);

void second_integration(unsigned int np, float dt, float4* v, float4* f, unsigned int block_size);
