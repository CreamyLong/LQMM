#pragma once

#include <cuda_runtime.h> //float3 ,float4

void compute_info(unsigned int np, float4* v, float4* f, float2* scratch, float* info, unsigned int block_size);
