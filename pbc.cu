#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
//#include <sys/time.h> //only windows
#include <time.h>
#include <iostream>

#include <cuda_runtime.h>
#include "device_functions.h"
#include <device_launch_parameters.h>


 // implement periodic bondary condition
__device__ __host__  float pbc(float x, float box_len)
{
	float box_half = box_len * 0.5;
	if (x > box_half) x -= box_len;
	else if (x < -box_half) x += box_len;
	return x;
}