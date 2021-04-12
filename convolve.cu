#include "convolve.h"

__global__ void _convolve(float* img, int img_w, int img_h, float* kernel, int kernel_w, int kernel_h, float* output) {
}

template<int kernel_w, int kernel_h>
 void Convolve<kernel_w, kernel_h>::doConvolve(float* img, int img_w, int img_h, float* output) {
	// Requires that img and output are cudaMalloc'd by caller
	dim3 blockSize(32, 32);
	// Make Gridsize
	dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
                 
	_convolve<<<gridDims, blockSize>>>(img, img_w, img_h, kernel, kernel_w, kernel_h, output);
}

//um wait holdon
//forward declare the templates we will need later
template class Convolve<3,3>;
