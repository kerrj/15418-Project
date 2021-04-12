#ifndef __CONVOLVE_H__
#define __CONVOLVE_H__
#include <cuda_runtime.h>
#include <cuda.h>

// TODO: img is readonly? 
__global__ void _convolve(float* img, int img_w, int img_h, float* kernel, int kernel_w, int kernel_h, float* output);

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
