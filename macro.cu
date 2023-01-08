#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <time.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <algorithm>
#include "utils.h"



/*system information collection for temperature, kinetic energy, potentialand total energy*/
__global__ void compute_info_sums_kernel(
    unsigned int np,
    float4* v,
    float4* f,
    float2* scratch)
{


    /*
        若线程频繁对某个数据进行读写操作，可以设置操作的数据常驻缓存，
        这样可以进一步提高代码的运行效率，并且同一个线程块内的所有线程共享该内存区域。
    */
    extern __shared__ float2 sdata[];   

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float2 tempo = make_float2(0.0, 0.0);
    if (i < np)
    {
        float4 vi = v[i];
        float mass = vi.w;
        tempo.x = mass * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
        tempo.y = f[i].w;

        if (i + blockDim.x < np)
        {
            vi = v[i + blockDim.x];
            mass = vi.w;
            tempo.x += mass * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
            tempo.y += f[i + blockDim.x].w;
        }
    }

    sdata[threadIdx.x] = tempo;

    /*
        __syncthreads()将确保线程块中的每个线程都执行完 __syncthreads()前面的语句后，才会执行下一条语句。
    */
    __syncthreads();

    int offs = blockDim.x >> 1;
    while (offs > 0)
    {
        if (threadIdx.x < offs)
        {
            sdata[threadIdx.x].x += sdata[threadIdx.x + offs].x;
            sdata[threadIdx.x].y += sdata[threadIdx.x + offs].y;
        }
        offs >>= 1;
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        scratch[blockIdx.x].x = sdata[0].x;
        scratch[blockIdx.x].y = sdata[0].y;
    }
}


__global__ void compute_info_final_kernel(
    unsigned int np,
    float* info,
    float2* scratch,
    unsigned int num_partial_sums)
{
    extern __shared__ float2 sdata[];
    float2 final_sum = make_float2(0.0, 0.0);
    for (int start = 0; start < num_partial_sums; start += blockDim.x * 2)
    {
        float2 tempo = make_float2(0.0, 0.0);
        if (start + threadIdx.x < num_partial_sums)
        {
            float2 scr = scratch[start + threadIdx.x];
            tempo.x = scr.x;
            tempo.y = scr.y;
            if (start + threadIdx.x + blockDim.x < num_partial_sums)
            {
                scr = scratch[start + threadIdx.x + blockDim.x];
                tempo.x += scr.x;
                tempo.y += scr.y;
            }
        }

        sdata[threadIdx.x] = tempo;
        __syncthreads(); //	//对线程块中的线程进行同步


        int offs = blockDim.x >> 1;
        while (offs > 0)
        {
            if (threadIdx.x < offs)
            {
                sdata[threadIdx.x].x += sdata[threadIdx.x + offs].x;
                sdata[threadIdx.x].y += sdata[threadIdx.x + offs].y;
            }
            offs >>= 1;
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            final_sum.x += sdata[0].x;
            final_sum.y += sdata[0].y;
        }
    }

    if (threadIdx.x == 0)
    {
        float ekin = 0.5 * final_sum.x;
        float potential = 0.5 * final_sum.y;
        unsigned int nfreedom = 3 * np - 3;
        float temp = 2.0 * ekin / float(nfreedom);
        float energy = ekin + potential;

        info[0] = temp;
        info[1] = potential;
        info[2] = energy;
    }
}

void compute_info(
    unsigned int np,
    float4* v,
    float4* f,
    float2* scratch,
    float* info,
    unsigned int block_size)
{

    unsigned int n_blocks = (int)ceil((float)np / (float)block_size);
    dim3 grid(n_blocks, 1, 1);
    dim3 threads(block_size / 2, 1, 1);
    unsigned int shared_bytes = sizeof(float2) * block_size / 2;

    compute_info_sums_kernel << <grid, threads, shared_bytes >> > (np, v, f, scratch);
    // block until the device has completed
    //cudaThreadSynchronize();
    cudaDeviceSynchronize(); 	//Function used for waiting for all kernels to finish

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");

    int final_block_size = 512;
    grid = dim3(1, 1, 1);
    threads = dim3(final_block_size, 1, 1);
    shared_bytes = sizeof(float2) * final_block_size;

    compute_info_final_kernel << <grid, threads, shared_bytes >> > (np, info, scratch, n_blocks);
    // block until the device has completed
    
    cudaDeviceSynchronize(); 	//Function used for waiting for all kernels to finish


    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
}