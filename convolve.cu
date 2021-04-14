#include "convolve.h"

__global__ void _convolve(const float* __restrict__ img, int img_w, int img_h,const float* __restrict__ kernel, int kernel_w, int kernel_h, float* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x - (kernel_w/2) < 0 || x + (kernel_w/2) >= img_w) return;
	if(y - (kernel_h/2) < 0 || y + (kernel_h/2) >= img_h) return;
	
	float val = 0.0;
	for(int r = 0; r < kernel_h; r++) {
		int newy = y + r - (kernel_h / 2);
		for(int c = 0; c < kernel_w; c++) {
			int newx = x + c - (kernel_w / 2);
			val += kernel[r*kernel_w + c] * img[newy*img_w + newx];
		}
	}
	output[img_w * y + x] = val;
}

template<int kernel_w, int kernel_h>
 void Convolve<kernel_w, kernel_h>::doConvolve(float* img, int img_w, int img_h, float* output) {
	// Requires that img and output are cudaMalloc'd by caller
	const dim3 blockSize(32, 32);
	// Make Gridsize
	const dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
                 
	_convolve<<<gridDims, blockSize>>>(img, img_w, img_h, kernel, kernel_w, kernel_h, output);
}

//um wait holdon
//forward declare the templates we will need later
template class Convolve<3,3>;
template class Convolve<5,5>;
template class Convolve<5,1>;
template class Convolve<1,5>;
template class Convolve<9,1>;
template class Convolve<1,9>;
template class Convolve<3,1>;
template class Convolve<1,3>;
