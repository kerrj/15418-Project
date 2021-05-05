#ifndef __CONVOLVE_H__
#define __CONVOLVE_H__
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void _convolve(const float* __restrict__ img, int img_w, int img_h, const float* kernel, int kernel_w, int kernel_h, float* __restrict__ output);



class Convolve {
private:
	int bufNum;
	int kernel_w,kernel_h;
public:
	Convolve(){}
	Convolve(float* input_kernel, int kernel_w, int kernel_h, int bnum);
	
	
	void doConvolve(float* img, int img_w, int img_h, float* output);

};
#endif
