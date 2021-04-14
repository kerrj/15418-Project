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
	if(x - (nmswindow/2) < 0 || x + (nmswindow/2) >= img_w) return;
	if(y - (nmswindow/2) < 0 || y + (nmswindow/2) >= img_h) return;
	
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
void scan_down_kernel(unsigned short* device_data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if index is going to be written to at this level
    if(index % size == size - 1) {
        // Get the index it will be combined with (halfway) 
        int j = index - (size >> 1);
        device_data[index] = device_data[index] + device_data[j];
    }
}

__global__
void scan_up_kernel(unsigned short* device_data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index % size == size - 1) {
        int j = index - (size >> 1);
        unsigned short original = device_data[index];
        device_data[index] += device_data[j];
        device_data[j] = original;
    }
}



__global__ void _collapse(const unsigned short* __restrict__ scanResult, const float* __restrict__ activations, int size, unsigned short* __restrict__ locations, float* __restrict__ outputActivations) {
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
    /* Taken from assignment 2 */
    const int threadsPerBlock = 128;
    const int blocks = (nextPow2(length) + threadsPerBlock - 1) / threadsPerBlock;

    int i = 2;
    for (; i < nextPow2(length); i=(i<<1)) {
        scan_down_kernel<<<blocks, threadsPerBlock>>>(device_data, i);
    }
    unsigned short x = 0;
    cudaMemcpy(&device_data[nextPow2(length)-1], &x, sizeof(short), cudaMemcpyHostToDevice);
    for(; i > 1; i=(i>>1)) {
        scan_up_kernel<<<blocks, threadsPerBlock>>>(device_data, i);
    }
}


void collapse(const unsigned short* scanResult, const float* activations, int size, unsigned short* locations, float* outputActivations) {
	const int threadsPerBlock = 128;
	const int blocks = size;
	_collapse<<<blocks, threadsPerBlock>>>(scanResult, activations, size, locations, outputActivations);
	
}
