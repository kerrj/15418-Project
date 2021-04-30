#ifndef __HARRIS_H__
#define __HARRIS_H__
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <algorithm>

void harris(float* gradX, float* gradY, int img_w, int img_h, float* activations, unsigned short* threshold_output);

void scan(unsigned short* output, int size);

void collapse(const unsigned short* scanResult, const float* activations, int size, unsigned int* locations, float* outputActivations);

std::vector<unsigned int> selectCorners(unsigned int* locations, float* outputActivations, int numCorners, int numSelect);
#endif


