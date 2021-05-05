#include "convolve.h"
#include <vector>
const int BDIM=32;
__constant__ float buf1[25];
__constant__ float buf2[25];
__constant__ float buf3[25];
__global__ void _convolve1(const float* __restrict__ img, const int img_w, const int img_h, const int kernel_w, const int kernel_h, float* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// kernel_w >> 1 is that same as kernel_w/2
	if(x - (kernel_w >> 1) < 0 || x + (kernel_w >> 1) >= img_w ||
		y - (kernel_h >> 1) < 0 || y + (kernel_h >> 1) >= img_h){
		return;	 
	}
	float val = 0.0;
	for(int r = 0; r < kernel_h; r++) {
		const int newy = y + r - (kernel_h >> 1);
		for(int c = 0; c < kernel_w; c++) {
			const int newx = x + c - (kernel_w >> 1);
			val += buf1[r*kernel_w + c] * img[newy*img_w + newx];
		}
	}
	output[img_w * y + x] = val;
}
__global__ void _convolve2(const float* __restrict__ img, const int img_w, const int img_h, const int kernel_w, const int kernel_h, float* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	// kernel_w >> 1 is that same as kernel_w/2
	if(x - (kernel_w >> 1) < 0 || x + (kernel_w >> 1) >= img_w ||
		y - (kernel_h >> 1) < 0 || y + (kernel_h >> 1) >= img_h){
		return;	 
	}
	float val = 0.0;
	for(int r = 0; r < kernel_h; r++) {
		const int newy = y + r - (kernel_h >> 1);
		for(int c = 0; c < kernel_w; c++) {
			const int newx = x + c - (kernel_w >> 1);
			val += buf2[r*kernel_w + c] * img[newy*img_w + newx];
		}
	}
	output[img_w * y + x] = val;
}

__global__ void _convolve3(const float* __restrict__ img, const int img_w, const int img_h, const int kernel_w, const int kernel_h, float* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	// kernel_w >> 1 is that same as kernel_w/2
	if(x - (kernel_w >> 1) < 0 || x + (kernel_w >> 1) >= img_w ||
		y - (kernel_h >> 1) < 0 || y + (kernel_h >> 1) >= img_h){
		return;	 
	}
	float val = 0.0;
	for(int r = 0; r < kernel_h; r++) {
		const int newy = y + r - (kernel_h >> 1);
		for(int c = 0; c < kernel_w; c++) {
			const int newx = x + c - (kernel_w >> 1);
			val += buf3[r*kernel_w + c] * img[newy*img_w + newx];
		}
	}
	output[img_w * y + x] = val;
}


 void Convolve::doConvolve(float* img, int img_w, int img_h, float* output) {
	// Requires that img and output are cudaMalloc'd by caller
	const dim3 blockSize(BDIM, BDIM);
	// Make Gridsize
	const dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
	
	switch(bufNum){
		case 0:
			_convolve1<<<gridDims, blockSize>>>(img, img_w, img_h, kernel_w, kernel_h, output);
			break;
		case 1:
			_convolve2<<<gridDims, blockSize>>>(img, img_w, img_h, kernel_w, kernel_h, output);
			break;
		case 2:
			_convolve3<<<gridDims, blockSize>>>(img, img_w, img_h, kernel_w, kernel_h, output);
			break;
	}
}


Convolve::Convolve(float* input_kernel, int w, int h,int bnum){
	kernel_w=w;
	kernel_h=h;
	bufNum=bnum;
	switch(bufNum){
		case 0:
			cudaMemcpyToSymbol(buf1, input_kernel, sizeof(float)*kernel_w*kernel_h);
			break;
		case 1:
			cudaMemcpyToSymbol(buf2, input_kernel, sizeof(float)*kernel_w*kernel_h);
			break;
		case 2:
			cudaMemcpyToSymbol(buf3, input_kernel, sizeof(float)*kernel_w*kernel_h);
			break;
	}
}
