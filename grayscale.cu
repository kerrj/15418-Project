#include "grayscale.h"
#include <stdio.h>

__global__ void _im2gray(const uint8_t* __restrict__ data, int size, float* __restrict__ output) {
	const size_t ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind<size)output[ind]= data[ind]/255.0;
}
void im2gray(uint8_t* data, int size, float* output,dim3 gridSize, dim3 blockSize) {
	_im2gray<<<gridSize,blockSize>>>(data,size,output);
}
