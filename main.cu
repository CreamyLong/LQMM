#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h> //make_float3

#include "init.h"
#include "utils.h"
#include "move.h"
#include "force.h"
#include "macro.h"
#include "thermostats.cuh"


// main function
int run()
{
	/* phase seperation parameters for video:
	unsigned int nsteps = 200000;
	unsigned int nprint = 5000;
	float3 epsilon = make_float3(1.0, 0.2, 1.0);
	*/

	//running parameters

	/*
	unsigned的作用就是将数字类型无符号化， 例如 int 型的范围：-2^31 ~ 2^31 - 1，而unsigned int的范围：0 ~ 2^32。
	无符号版本和有符号版本的区别就是无符号类型能保存2倍于有符号类型的正整数数据。
	*/
	unsigned int np = 2700;                  // the number of particles
	unsigned int nsteps = 500;             // the number of time steps
	float dt = 0.001;                       // integration time step
	float rcut = 3.0;                       // the cutoff radius of interactions
	float temperature = 1.0;                // target temperature
	unsigned int nprint = 100;              //  period for data output
	unsigned int block_size = 128;          //  the number of threads in a block           

	clock_t start, end; // start time , end time

	
	
	float3 box = make_float3(15.0, 15.0, 15.0);     // box size in x, y, and z directions
	
	//Lennard-Jones兰纳-琼斯连续势能计算，是用来模拟两个电中性的分子或原子间相互作用势能的一个比较简单的数学模型。
	/*
	    epsilon等于势能阱的深度
		epsilon.x for type 1.0 and 1.0;
		epsilon.y for type 1.0 and 2.0;
		epsilon.z for type 1.0 and 2.0
	*/
	float3 epsilon = make_float3(1.0, 0.5, 1.0);   
	
	/*
	    sigma是互相作用的势能正好为零时的两体距离
		sigma.x for type 1.0 and 1.0;
		sigma.y for type 1.0 and 2.0;
		sigma.z for type 1.0 and 2.0
	*/
	float3 sigma = make_float3(1.0, 1.0, 1.0);     // 
	
	float min_dis = sigma.x * 0.9;	// the minimum distance between particles for system generation
	
	float3 lj1, lj2;

	//第一项可认为是对应于两体在近距离时以互相排斥为主的作用
	lj1.x = 4.0 * epsilon.x * pow(sigma.x, float(12.0));
	lj1.y = 4.0 * epsilon.y * pow(sigma.y, float(12.0));
	lj1.z = 4.0 * epsilon.z * pow(sigma.z, float(12.0));

	//第二项对应两体在远距离以互相吸引（例如通过范德瓦耳斯力）为主的作用
	lj2.x = 4.0 * epsilon.x * pow(sigma.x, float(6.0));
	lj2.y = 4.0 * epsilon.y * pow(sigma.y, float(6.0));
	lj2.z = 4.0 * epsilon.z * pow(sigma.z, float(6.0));

	//host memory allocation
	float4* h_r = (float4*)malloc(np * sizeof(float4));  // rx, ry, rz, type(0, 1, 2 ...)
	float4* h_v = (float4*)malloc(np * sizeof(float4));  // vx, vy, vz, mass
	float4* h_f = (float4*)malloc(np * sizeof(float4));  // fx, fy, fz, potential
	float* h_info = (float*)malloc(16 * sizeof(float));  // temperature, potential, energy ...


	//device memory allocation
	float4* d_r;
	float4* d_v;
	float4* d_f;
	float* d_info;
	float2* d_scratch;

	cudaMalloc((void**)&d_r, np * sizeof(float4));     // rx, ry, rz, type(0, 1, 2 ...)
	cudaMalloc((void**)&d_v, np * sizeof(float4));     // vx, vy, vz, mass
	cudaMalloc((void**)&d_f, np * sizeof(float4));     // fx, fy, fz, potential
	cudaMalloc((void**)&d_info, 16 * sizeof(float));	  // temperature, potential, energy ...
	cudaMalloc((void**)&d_scratch, (np / block_size + 1) * sizeof(float2));	  // temporary data ...

	FILE* traj = fopen("traj.xyz", "w");                 // trajectory file in XYZ format that can be open by VMD

	/* generate system information */

	printf("Starting simulation with %d atoms for %d steps.\n", np, nsteps);
	printf("Generating system.\n", np, nsteps);

	init(np, h_r, h_v, box, min_dis);
	
	const int NoseHoverChain = 3; //NH链长度
	double tauT = 0.1;//粒子热浴温度设置
	double eta_mass[NoseHoverChain], eta_dot[NoseHoverChain], eta_dotdot[NoseHoverChain];
	double nhc_factor = 1.0;
	//==================NHC=====================
	eta_mass[0] = 3.0 * np * temperature * pow(tauT, float(2.0));
	for (int ich = 1; ich < NoseHoverChain; ich++) {
		eta_mass[ich] = temperature * pow(tauT, float(2.0));
	}
	memset(eta_dot, '\0', (NoseHoverChain) * sizeof(double));
	memset(eta_dotdot, '\0', NoseHoverChain * sizeof(double));

	for (int ith = 0; ith < NoseHoverChain; ith++) {
		eta_dot[ith] = R2S() - 0.5;
		double T0 = eta_mass[ith] * pow(eta_dot[ith], float(2.0));
		double scale = sqrt(temperature / T0);

		eta_dot[ith] *= scale;
	}

	//calculate thermostat force
	for (int ich = 1; ich < NoseHoverChain; ich++) {
		eta_dotdot[ich] = (eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - temperature) / eta_mass[ich];
	}
	
	double* d_eta_dotdot;
	double* d_eta_dot;
	double* d_eta_mass;

	cudaMalloc((void**)&d_eta_dotdot, NoseHoverChain * sizeof(double));    
	cudaMalloc((void**)&d_eta_dot, NoseHoverChain * sizeof(double));     
	cudaMalloc((void**)&d_eta_mass, NoseHoverChain * sizeof(double));     

	cudaMemcpy(d_eta_dotdot, eta_dotdot, NoseHoverChain * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eta_dot, eta_dot, NoseHoverChain * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eta_mass, eta_dotdot, NoseHoverChain * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_r, h_r, np * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v, np * sizeof(float4), cudaMemcpyHostToDevice);

	start = clock();

	/* main MD loop */
	printf("Running simulation.\n", np, nsteps);

	for (unsigned int step = 0; step <= nsteps; step++) //running simulation loop
	{
		first_NHC(np, dt, box, d_r, d_v, d_f, d_info, temperature, nhc_factor, d_eta_dotdot, d_eta_dot, d_eta_mass, NoseHoverChain);

		/* first integration for velverlet */
		first_integration(np, dt, box, d_r, d_v, d_f, block_size);

		/* force calculation */
		force_calculation(np, box, lj1, lj2, d_r, d_f, rcut * rcut, block_size);

		/* temperature and potential calculation */
		compute_info(np, d_v, d_f, d_scratch, d_info, block_size);

		/* second integration for velverlet */
		second_integration(np, dt, d_v, d_f, block_size);

		second_NHC(np, dt, box, d_r, d_v, d_f, d_info, temperature, nhc_factor, d_eta_dotdot, d_eta_dot, d_eta_mass, NoseHoverChain);

		/* write output frames and system information, if requested */
		if ((step % nprint) == 0)
		{
			cudaMemcpy(h_r, d_r, np * sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_info, d_info, 16 * sizeof(float), cudaMemcpyDeviceToHost);
			output(traj, step, h_info, h_r, np);
			printf("time step %d \n", step);
		}
	}

	end = clock();
	double  duration = (double)(end - start) / CLOCKS_PER_SEC;

	printf("%f seconds\n", duration);

	fclose(traj);
	free(h_r);
	free(h_v);
	free(h_f);
	free(h_info);

	cudaFree(d_r);
	cudaFree(d_v);
	cudaFree(d_f);
	cudaFree(d_info);
	return 0;
}

int main(int argc, char** argv) {
	run();
}

