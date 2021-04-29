#include "harris.h"

#include <stdio.h>

const int window = 9;


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void _harrisActivation(const float* __restrict__ gradX, const float* __restrict__ gradY, int img_w, int img_h, float* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x - (window/2) < 0 || x + (window/2) >= img_w) return;
	if(y - (window/2) < 0 || y + (window/2) >= img_h) return;
	
	float3 val = make_float3(0, 0, 0); // Ixx, Iyy, Ixy
	for(int r = 0; r < window; r++) {
		int newy = y + r - (window / 2);
		for(int c = 0; c < window; c++) {
			int newx = x + c - (window / 2);
			const float gX = gradX[newy*img_w + newx];
			const float gY = gradY[newy*img_w + newx];
			val.x += gX * gX;
			val.y += gY * gY;
			val.z += gX * gY;
		}
	}
	// Compute R = det(M) - k * tr(M)^2
	const float det = val.x * val.y - val.z * val.z;
	const float R=det - 0.05 * (val.x + val.y) * (val.x + val.y);	
	if(R>0.3){
		output[img_w * y + x] = R;
	}else{
		output[img_w * y + x] = 0;
	}
}

const int nmswindow=3;
__global__ void _nms(const float* __restrict__ activations, int img_w, int img_h, unsigned short* __restrict__ threshold_output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	threshold_output[y*img_w + x] = 0;
	if(x - (window/2) < 0 || x + (window/2) >= img_w) return;
	if(y - (window/2) < 0 || y + (window/2) >= img_h) return;
	
	const float val = activations[y*img_w + x];
	if(val < 0.001) return;
	for(int r = 0; r < nmswindow; r++) {
		int newy = y + r - (nmswindow / 2);
		for(int c = 0; c < nmswindow; c++) {
			int newx = x + c - (nmswindow / 2);
			const float newVal = activations[newy*img_w + newx];
			// Returns since value is 0 if there exists a higher-activation neighbor
			if(newVal > val) {
				return;
			}
		}
	}
	threshold_output[y*img_w + x] = 1;
}

__global__
void scan_down_kernel(unsigned short* device_data, int size, int device_data_size_of_array_yes_sir_not_the_previous_size) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x+1) * size - 1;
    // Check if index is going to be written to at this level
    if(index < device_data_size_of_array_yes_sir_not_the_previous_size) {
        // Get the index it will be combined with (halfway) 
        const int j = index - (size >> 1);
        device_data[index] = device_data[index] + device_data[j];
    }
}

__global__
void scan_up_kernel(unsigned short* device_data, int size, int device_data_size_of_array_yes_sir_not_the_previous_size) {
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = (blockIdx.x * blockDim.x + threadIdx.x+1) * size - 1;
    // if(index % size == size - 1) {
    if(index<device_data_size_of_array_yes_sir_not_the_previous_size){
        const int j = index - (size >> 1);
        const short original = device_data[index];
        device_data[index] += device_data[j];
        device_data[j] = original;
    }
}


__global__ void _collapse(const unsigned short* __restrict__ scanResult, const float* __restrict__ activations, int size, unsigned int* __restrict__ locations, float* __restrict__ outputActivations) {
	const int index = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if(index>=size)return;
    if(scanResult[index]>scanResult[index-1]){
        //this is a new index to count in output
        locations[scanResult[index-1]] = index-1;
        // printf("ScanRes: %hu, %hu\n", scanResult[index-1]/1280,scanResult[index-1]%1280 );
        outputActivations[scanResult[index-1]] = activations[scanResult[index-1]];
    }
}

void harris(float* gradX, float* gradY, int img_w, int img_h, float* activations, unsigned short* threshold_output) {
	// Requires that img and output are cudaMalloc'd by caller
	const dim3 blockSize(32, 32);
	// Make Gridsize
	const dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
                 
                 
	_harrisActivation<<<gridDims, blockSize>>>(gradX, gradY, img_w, img_h, activations);
	cudaDeviceSynchronize();
	_nms<<<gridDims, blockSize>>>(activations, img_w, img_h, threshold_output);
}

void scan(unsigned short* device_data, int length)
{
    const int threadsPerBlock = 128;
    const int length_nextPow2 = nextPow2(length);
    int i = 2;
    for (; i < length_nextPow2; i=(i<<1)) {
        const size_t n = length_nextPow2/i;//num elements to compute
        const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        scan_down_kernel<<<blocks, threadsPerBlock>>>(device_data, i, length_nextPow2);
    }
    const short x = 0;
    cudaMemcpy(&device_data[length_nextPow2-1], &x, sizeof(short), cudaMemcpyHostToDevice);
    for(; i > 1; i=(i>>1)) {
        const size_t n = length_nextPow2/i;//num elements to compute
        const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        //const int blocks = (length_nextPow2 + threadsPerBlock - 1) / threadsPerBlock;
        scan_up_kernel<<<blocks, threadsPerBlock>>>(device_data, i,length_nextPow2);
    }
}


void collapse(const unsigned short* scanResult, const float* activations, int size, unsigned int* locations, float* outputActivations) {
	const int threadsPerBlock = 128;
	const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
	_collapse<<<blocks, threadsPerBlock>>>(scanResult, activations, size, locations, outputActivations);
	
}
