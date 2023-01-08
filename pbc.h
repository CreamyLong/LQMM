#pragma once
#include <cuda_runtime.h> //float3 ,float4

__host__ __device__  float pbc(float x, float box_len);
