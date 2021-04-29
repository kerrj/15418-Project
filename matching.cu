#include "matching.h"
#include "brief.h"

__device__ int numberOfSetBits(char c)
{
     int numBits = 0;
     for(int i = 0; i < 8; i++) {
     	numBits += ((1 << i) & c) >> i;
     }
     
     return numBits;
}
__device__ int numberOfSetBitsLong(long v)
{
	//TODO make this
	return 42;
}
__global__ void _makeDistMatrix(const char* featureBuf1, const char* featureBuf2, int size1, int size2, FeatureDist* output, FeatureDist* outputTranspose){
	// Matrix is size2 rows by size1 columns. 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x>=size1 || y>=size2)return;
	// Get the feature from location x and y
	int diff = 0;
	for(int i = 0; i < CHARS_PER_BRIEF; i++) {
		const char f1 = featureBuf1[CHARS_PER_BRIEF * x + i];
		const char f2 = featureBuf2[CHARS_PER_BRIEF * y + i];
		diff += 8 - numberOfSetBits(f1 ^ f2);//TODO make this use 64 bit ints
	}
	output[x + size1 * y].distance = diff;
	output[x + size1 * y].f2Index = y;
	output[x + size1 * y].f1Index = x;
	
	outputTranspose[y + size2 * x].distance = diff;
	outputTranspose[y + size2 * x].f2Index = y;
	outputTranspose[y + size2 * x].f1Index = x;
}

void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, FeatureDist *output, FeatureDist* outputTranspose){
	
	const dim3 blockSize(32, 32);
	// Make Gridsize
	const dim3 gridDims((size1 + blockSize.x - 1) / blockSize.x,
                 (size2 + blockSize.y - 1) / blockSize.y);
                 
	_makeDistMatrix<<< gridDims, blockSize >>>(featureBuf1,featureBuf2,size1,size2,output, outputTranspose);
}
