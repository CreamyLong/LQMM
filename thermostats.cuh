#pragma once
#include <cuda_runtime.h> //float3 ,float4

void first_NHC(
	unsigned int np,
	float dt,
	float3 box,
	float4* r,
	float4* v,
	float4* f,
	float* info,
	float temperature,
	double nhc_factor,
	double* eta_dotdot,
	double* eta_dot,
	double* eta_mass,
	int NoseHoverChain
);

void second_NHC(
	unsigned int np,
	float dt,
	float3 box,
	float4* r,
	float4* v,
	float4* f,
	float* info,
	float temperature,
	double nhc_factor,
	double* eta_dotdot,
	double* eta_dot,
	double* eta_mass,
	int NoseHoverChain
);