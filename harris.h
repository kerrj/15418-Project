#ifndef __HARRIS_H__
#define __HARRIS_H__
#include <cuda_runtime.h>
#include <cuda.h>


void harris(float* gradX, float* gradY, int img_w, int img_h, float* activations, unsigned short* threshold_output);

void scan(unsigned short* output, int size);

void collapse(const unsigned short* scanResult, const float* activations, int size, unsigned int* locations, float* outputActivations);

#endif


