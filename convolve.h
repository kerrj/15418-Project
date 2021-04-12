#ifndef __CONVOLVE_H__
#define __CONVOLVE_H__
#include <cuda_runtime.h>
#include <cuda.h>

// TODO: img is readonly? 
__global__ void _convolve(const float* __restrict__ img, int img_w, int img_h, const float* __restrict__ kernel, int kernel_w, int kernel_h, float* __restrict__ output);

template<int kernel_w, int kernel_h>
class Convolve {
private:
	float* kernel;

public:
	Convolve(){}
	Convolve(float* input_kernel){
		cudaMalloc(&kernel, sizeof(float)*kernel_w*kernel_h);
		cudaMemcpy(kernel, input_kernel, sizeof(float)*kernel_w*kernel_h,cudaMemcpyHostToDevice);
	}
	
	~Convolve() {
		cudaFree(&kernel);
	}
	
	void doConvolve(float* img, int img_w, int img_h, float* output);

};
#endif
