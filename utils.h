#pragma once

void checkCUDAError(const char* msg);

void output(FILE* traj, unsigned int step, float* info, float4* r, unsigned int np);