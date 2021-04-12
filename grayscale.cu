#include "grayscale.h"
#include <stdio.h>
__global__ void _im2gray(uint8_t* data, int size, float* output) {
}
void im2gray(uint8_t* data, int size, float* output,dim3 gridSize, dim3 blockSize) {
	_im2gray<<<gridSize,blockSize>>>(data,size,output);
	cudaDeviceSynchronize();
}
