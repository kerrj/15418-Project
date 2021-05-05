#include "harris.h"

#include <stdio.h>
#include <chrono>
#include <string>
#include <cuda_fp16.h>
const int window = 9;
const int halfwin=window/2;

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
const int BDIM = 32;
template<typename T>
inline __device__ T& block2Buf(T* blockBuf, const int x, const int y, const int pad_size){
	return blockBuf[pad_size + x + (pad_size + y)*(BDIM+pad_size*2)];
}
inline __device__ float getVal(const half* blockBuf,const float* im,const int im_w, const int x,const int y,const int pad_size){
	if(x<0 || y<0 || x>=BDIM || y>=BDIM){
		return im[(blockDim.y*blockIdx.y+y)*im_w + blockDim.x*blockIdx.x+x];
	}else{
		return __half2float(blockBuf[pad_size + x + (pad_size + y)*(BDIM+pad_size*2)]);
	}
}
inline __device__ void fillSquare(half* blockBuf, const float* im, const int x, const int y, const int pad_size, const int im_w, const int im_h,
		const int globalX,const int globalY){
	for(int dy = 0; dy < pad_size; dy++) {
		if(globalY + dy >= im_h || globalY + dy < 0) continue;
		for(int dx = 0; dx < pad_size; dx++) {
			if(globalX + dx >= im_w || globalX + dx < 0) continue;
			block2Buf(blockBuf, x + dx, y + dy, pad_size) = __float2half(im[globalX + dx + (globalY + dy)*im_w]);
		}
	}
}

inline __device__ void fillBuf(half* blockBuf,const float* im, const int im_w, const int im_h, const int pad_size, int globalX, int globalY){
	//pad size is the size of the border we need to load in (should be window/2 for kernels)
	//we guarantee that globalX and globalY will be in bounds
	//first fill the middle
	block2Buf(blockBuf,threadIdx.x,threadIdx.y,pad_size) = __float2half(im[globalX + im_w*globalY]);
	//then handle the edges
	
	bool corner=false;
	int tlx,tly;
	if(threadIdx.x == 0){
		//left col
		if(threadIdx.y==0){
			//top left
			corner=true;
			tlx=-pad_size;
			tly=-pad_size;
		}
		for(int i = -1;i >= -pad_size;i--){
			if(globalX+i<0)break;
			block2Buf(blockBuf,threadIdx.x + i,threadIdx.y,pad_size) = __float2half(im[globalX + i + globalY*im_w]);
		}
	}else if(threadIdx.y == 0){
		//top row
		if(threadIdx.x==blockDim.x-1){
			//top right
			corner=true;
			tlx=blockDim.x;
			tly=-pad_size;
		}
		for(int i = -1;i >= -pad_size;i--){
			if(globalY+i<0)break;
			block2Buf(blockBuf,threadIdx.x,threadIdx.y+i,pad_size) = __float2half(im[globalX + (globalY + i)*im_w]);
		}
	}else if(threadIdx.x == blockDim.x-1){
		//right col
		if(threadIdx.y==blockDim.y-1){
			//bot right
			corner=true;
			tlx=blockDim.x;
			tly=blockDim.y;
		}
		for(int i = 1;i <= pad_size;i++){
			if(globalX+i>=im_w)break;
			block2Buf(blockBuf,threadIdx.x + i,threadIdx.y,pad_size) = __float2half(im[globalX + i + globalY*im_w]);
		}
	}else if(threadIdx.y == blockDim.y-1){
		//bot row
		if(threadIdx.x==0){
			//bot left
			corner=true;
			tlx=-pad_size;
			tly=blockDim.y;
		}
		for(int i = 1;i <= pad_size;i++){
			if(globalY+i>=im_h)break;
			block2Buf(blockBuf,threadIdx.x,threadIdx.y+i,pad_size) = __float2half(im[globalX + (globalY + i)*im_w]);
		}
	}
	if(corner){
		fillSquare(blockBuf,im,tlx,tly,pad_size,im_w,im_h,globalX,globalY);
	}
	
}
const int BUFSIZE=(BDIM + 2*halfwin)*(BDIM + 2*halfwin);
__global__ void _harrisActivation(const float* __restrict__ gradX, const float*  __restrict__ gradY,const int img_w,const int img_h, float* __restrict__ output) {
	__shared__ half gradXBuf[BUFSIZE];
	__shared__ half gradYBuf[BUFSIZE];
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x - halfwin < 0 || x + halfwin >= img_w 
		|| y - halfwin < 0 || y + halfwin >= img_h){
		 __syncthreads();
		 return;
	}
	fillBuf(gradXBuf,gradX,img_w,img_h,halfwin,x,y);
	fillBuf(gradYBuf,gradY,img_w,img_h,halfwin,x,y);
	__syncthreads();
	
	half valx=0,valy=0,valz=0; // Ixx, Iyy, Ixy
	const int tilew=(BDIM+halfwin*2);
	for(int r = 0; r < window; r++) {
		//const int newy = y + r - halfwin;
		const int newy = threadIdx.y+r-halfwin;
		for(int c = 0; c < window; c++) {
			//int newx = x + c - halfwin;
			//const float gX = gradX[newy*img_w + newx];
			//const float gY = gradY[newy*img_w + newx];
			
			const int newx = threadIdx.x+c-halfwin;
			const int bufid = halfwin + newx + (halfwin + newy)*tilew;
			const half gX = (gradXBuf[bufid]);
			const half gY = (gradYBuf[bufid]);
			
			//const float gX = getVal(gradXBuf,gradX,img_w,newx,newy,halfwin);
			//const float gY = getVal(gradYBuf,gradY,img_w,newx,newy,halfwin);
			valx += gX * gX;
			valy += gY * gY;
			valz += gX * gY;
		}
	}
	// Compute R = det(M) - k * tr(M)^2
	const float det = __half2float(valx * valy - valz * valz);
	const float R=det - 0.05 * __half2float((valx + valy) * (valx + valy));
	const int keepflag = R>0.3;
	output[img_w * y + x] = R*keepflag;
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
        outputActivations[scanResult[index-1]] = activations[scanResult[index-1]];
    }
}
// Timing 
std::chrono::time_point<std::chrono::system_clock> tic2(){
	return std::chrono::system_clock::now();
}
double toc2(std::string msg,std::chrono::time_point<std::chrono::system_clock> & t){
	std::chrono::duration<double> elapsed=std::chrono::system_clock::now()-t;
	printf("Time for %s: %f\n",msg.c_str(),elapsed.count());
	t = tic2();
	return elapsed.count();
}
void harris(float* gradX, float* gradY, int img_w, int img_h, float* activations, unsigned short* threshold_output) {
	// Requires that img and output are cudaMalloc'd by caller
	const dim3 blockSize(BDIM, BDIM);
	// Make Gridsize
	const dim3 gridDims((img_w + blockSize.x - 1) / blockSize.x,
                 (img_h + blockSize.y - 1) / blockSize.y);
    auto t=tic2();
	_harrisActivation<<<gridDims, blockSize>>>(gradX, gradY, img_w, img_h, activations);
	cudaDeviceSynchronize();
	toc2("harris kern",t);
	_nms<<<gridDims, blockSize>>>(activations, img_w, img_h, threshold_output);
}

void scan(unsigned short* device_data, int length)
{
    const int threadsPerBlock = 64;
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
	const int threadsPerBlock = 64;
	const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
	_collapse<<<blocks, threadsPerBlock>>>(scanResult, activations, size, locations, outputActivations);	
}

bool compareCorners(std::pair<float, unsigned int> c1, std::pair<float, unsigned int> c2) {
	return c1.first > c2.first;
}

std::vector<unsigned int> selectCorners(unsigned int* locations, float* outputActivations, int numCorners, int numSelect) {
	// numCorners: Number of corners passed in
	// numSelect: Number of corners to return 
	// Sorts locations by outputActivations (outputActivations is unchanged)
	// Copy data from cuda memory
	std::vector<unsigned int> locations_local(numCorners);
	std::vector<float> activations_local(numCorners);
	cudaMemcpy(locations_local.data(), locations, sizeof(int)*numCorners, cudaMemcpyDeviceToHost);
	cudaMemcpy(activations_local.data(), outputActivations, sizeof(float)*numCorners, cudaMemcpyDeviceToHost);
	// Make pairs of (activation, location)
	std::vector<std::pair<float, unsigned int> > cornerPairs(numCorners);
	for(int i = 0; i < numCorners; i++)
		cornerPairs[i] = std::make_pair(activations_local[i], locations_local[i]);  
	// Do partial sort 
	std::partial_sort(cornerPairs.begin(), cornerPairs.begin() + numSelect, cornerPairs.end(), compareCorners);
	// Return locations only as a vector 
	for(int i = 0; i < numSelect; i++)
		locations_local[i] = cornerPairs[i].second;
	cudaMemcpy(locations, locations_local.data(), sizeof(int)*numSelect, cudaMemcpyHostToDevice);
	return locations_local;
};
