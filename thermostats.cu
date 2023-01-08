#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <cuda_runtime.h>
#include "device_functions.h"
#include <device_launch_parameters.h>
#include "utils.h"


extern "C" __global__ void first_NHC_kernel(
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
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	double dtArr[3];
	dtArr[0] = dt;
	dtArr[1] = dt / 2.0;
	dtArr[2] = dt / 4.0;

	double dof = 3 * np - 3.0;
	double akin = dof * info[0];
	double expfac = 1.0, factor_eta = 1.0;
	//=====================================================================
	eta_dotdot[0] = (akin - dof * temperature) / eta_mass[0];//calculate thermostat force
	eta_dot[NoseHoverChain - 1] += eta_dotdot[NoseHoverChain - 1] * 0.25 * dtArr[0];

	for (int ich = NoseHoverChain - 2; ich >= 0; ich--) {
		expfac = exp(-0.125 * dtArr[0] * eta_dot[ich + 1]);
		eta_dot[ich] *= expfac;
		eta_dot[ich] += eta_dotdot[ich] * 0.25 * dtArr[0];
		eta_dot[ich] *= expfac;
	}

	factor_eta *= exp(-0.5 * dtArr[0] * eta_dot[0]);
	akin *= factor_eta * factor_eta;

	eta_dotdot[0] = (akin - dof * temperature) / eta_mass[0];
	expfac = exp(-0.125 * dtArr[0] * eta_dot[1]);//thermostat eta_dot[0]
	eta_dot[0] *= expfac;
	eta_dot[0] += eta_dotdot[0] * 0.25 * dtArr[0];
	eta_dot[0] *= expfac;

	for (int ich = 1; ich < NoseHoverChain - 1; ich++) {
		expfac = exp(-0.125 * dtArr[0] * eta_dot[ich + 1]);
		eta_dot[ich] *= expfac;
		eta_dotdot[ich] = (eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - temperature) / eta_mass[ich];
		eta_dot[ich] += eta_dotdot[ich] * 0.25 * dtArr[0];
		eta_dot[ich] *= expfac;
	}
	eta_dotdot[NoseHoverChain - 1] = (eta_mass[NoseHoverChain - 2] * powf(eta_dot[NoseHoverChain - 2], float(2.0)) - temperature) / eta_mass[NoseHoverChain - 1];
	eta_dot[NoseHoverChain - 1] += eta_dotdot[NoseHoverChain - 1] * 0.25 * dtArr[0];
	//=====================================================================

	info[0] *= factor_eta * factor_eta;
	for (int ith = 0; ith < np; ith++) {
		v[ith].x *= factor_eta;
		v[ith].y *= factor_eta;
		v[ith].z *= factor_eta;
	}
	nhc_factor = factor_eta;
}

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
	int NoseHoverChain) 
{

	dim3 grid(1, 1, 1);
	dim3 block(NoseHoverChain, 1, 1);

	first_NHC_kernel << < grid, block >> > (np, dt, box, r, v, f, info, temperature, nhc_factor, eta_dotdot, eta_dotdot, eta_mass, NoseHoverChain);

	// block until the device has completed
	cudaDeviceSynchronize();


	// check if kernel execution generated an error
	checkCUDAError("kernel execution");
}


extern "C" __global__ void second_NHC_kernel(
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

)
{
	info[0] *= nhc_factor * nhc_factor;
	for (int ith = 0; ith < np; ith++) {
		v[ith].x *= nhc_factor;
		v[ith].y *= nhc_factor;
		v[ith].z *= nhc_factor;
	}

	double dtArr[3];
	dtArr[0] = dt;
	dtArr[1] = dt / 2.0;
	dtArr[2] = dt / 4.0;

	double dof = 3 * np - 3.0;
	double akin = dof * info[0];
	double expfac = 1.0, factor_eta = 1.0;

	eta_dotdot[0] = (akin - dof * temperature) / eta_mass[0];//calculate thermostat force
	eta_dot[NoseHoverChain - 1] += eta_dotdot[NoseHoverChain - 1] * 0.25 * dtArr[0];

	for (int ich = NoseHoverChain - 2; ich >= 0; ich--) {
		expfac = exp(-0.125 * dtArr[0] * eta_dot[ich + 1]);
		eta_dot[ich] *= expfac;
		eta_dot[ich] += eta_dotdot[ich] * 0.25 * dtArr[0];
		eta_dot[ich] *= expfac;
	}

	factor_eta *= exp(-0.5 * dtArr[0] * eta_dot[0]);
	akin *= factor_eta * factor_eta;

	eta_dotdot[0] = (akin - dof * temperature) / eta_mass[0];
	expfac = exp(-0.125 * dtArr[0] * eta_dot[1]);//thermostat eta_dot[0]
	eta_dot[0] *= expfac;
	eta_dot[0] += eta_dotdot[0] * 0.25 * dtArr[0];
	eta_dot[0] *= expfac;
	for (int ich = 1; ich < NoseHoverChain - 1; ich++) {
		expfac = exp(-0.125 * dtArr[0] * eta_dot[ich + 1]);
		eta_dot[ich] *= expfac;
		eta_dotdot[ich] = (eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - temperature) / eta_mass[ich];
		eta_dot[ich] += eta_dotdot[ich] * 0.25 * dtArr[0];
		eta_dot[ich] *= expfac;
	}
	eta_dotdot[NoseHoverChain - 1] = (eta_mass[NoseHoverChain - 2] * powf(eta_dot[NoseHoverChain - 2], float(2.0)) - temperature) / eta_mass[NoseHoverChain - 1];
	eta_dot[NoseHoverChain - 1] += eta_dotdot[NoseHoverChain - 1] * 0.25 * dtArr[0];

	nhc_factor = factor_eta;
}

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
	int NoseHoverChain)
{

	dim3 grid(1, 1, 1);
	dim3 block(NoseHoverChain, 1, 1);

	second_NHC_kernel << < grid, block >> > (np, dt, box, r, v, f, info, temperature, nhc_factor, eta_dotdot, eta_dotdot, eta_mass, NoseHoverChain);

	// block until the device has completed
	cudaDeviceSynchronize();


	// check if kernel execution generated an error
	checkCUDAError("kernel execution");
}