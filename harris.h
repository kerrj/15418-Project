#ifndef __HARRIS_H__
#define __HARRIS_H__
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void _harrisActivation(const float* __restrict__ gradX, const float* __restrict__ gradY, int img_w, int img_h, float* __restrict__ output);

__global__ void _nms(const float* __restrict__ activations, int img_w, int img_h, char* __restrict__ output); 
void harris(float* gradX, float* gradY, int img_w, int img_h, float* inter_buf, char* output);


#endif


