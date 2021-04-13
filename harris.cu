#include "harris.h"

const int window = 3;

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
	const float R=det - 0.06 * (val.x + val.y) * (val.x + val.y);	
	if(R>0.5){
		output[img_w * y + x] = R;
	}else{
		output[img_w * y + x] = 0;
	}
}

const int nmswindow=5;
__global__ void _nms(const float* __restrict__ activations, int img_w, int img_h, char* __restrict__ output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x - (nmswindow/2) < 0 || x + (nmswindow/2) >= img_w) return;
	if(y - (nmswindow/2) < 0 || y + (nmswindow/2) >= img_h) return;
	
	const float val = activations[y*img_w + x];
	
	// TODO: Figure out activation threshold	
	if(val < 0.001) {
		output[y*img_w + x] = 0;
		return; 
	}
	output[y*img_w + x] = 255; // Setting to 255 for visualization (could change to 1)
	for(int r = 0; r < nmswindow; r++) {
		int newy = y + r - (nmswindow / 2);
		for(int c = 0; c < nmswindow; c++) {
			int newx = x + c - (nmswindow / 2);
			const float newVal = activations[newy*img_w + newx];
			// Set to 0 if there exists a higher-activation neighbor
			if(newVal > val) {
				output[y*img_w + x] = 0;
				return;
			}
		}
	}
}

void harris(float* gradX, float* gradY, int img_w, int img_h, float* inter_buf, char* output) {
	// Requires that img and output are cudaMalloc'd by caller
	const dim3 blockSize(32, 32);
	// Make Gridsize
	const dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
                 
                 
	_harrisActivation<<<gridDims, blockSize>>>(gradX, gradY, img_w, img_h, inter_buf);
	_nms<<<gridDims, blockSize>>>(inter_buf, img_w, img_h, output);
}
