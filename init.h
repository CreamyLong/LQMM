#pragma once

#include <cuda_runtime.h> //float4

void init(unsigned int np, float4* r, float4* v, float3 box, float min_dis);
