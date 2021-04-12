#ifndef __GRAYSCALE__
#define __GRAYSCALE__

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void _im2gray(const uint8_t* __restrict__ data, int size, float* __restrict__ output);
void im2gray(uint8_t* data, int size, float* output,dim3 gridSize,dim3 blockSize);
#endif
