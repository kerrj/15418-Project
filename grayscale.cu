#include "grayscale.h"
#include <stdio.h>
__global__ void _im2gray(uint8_t* data, int size, float* output) {
	size_t ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= size ) return;
	output[ind]= (data[3*ind] + data[3*ind+1] + data[3*ind+2])/(765.0); //255 * 3
}
void im2gray(uint8_t* data, int size, float* output,dim3 gridSize, dim3 blockSize) {
	_im2gray<<<gridSize,blockSize>>>(data,size,output);
}
