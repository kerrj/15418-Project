#ifndef BRIEF_H
#define BRIEF_H
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include <vector>
const int CHARS_PER_BRIEF = 64;
const int BRIEF_SIZE = CHARS_PER_BRIEF * 8;
const int WINDOW = 9;

class Brief{

public:
	Brief() {
		// Make vectors
		std::default_random_engine generator;
  		std::uniform_int_distribution<int> distribution(-WINDOW/2,WINDOW/2);
  		
  		
		std::vector<int> vecpoints1_x(BRIEF_SIZE);
		std::vector<int> vecpoints1_y(BRIEF_SIZE);
		std::vector<int> vecpoints2_x(BRIEF_SIZE);
		std::vector<int> vecpoints2_y(BRIEF_SIZE);
  		for(int i=0; i<BRIEF_SIZE; i++) {
  			vecpoints1_x[i] = distribution(generator);
  			vecpoints1_y[i] = distribution(generator);
  			vecpoints2_x[i] = distribution(generator);
  			vecpoints2_y[i] = distribution(generator);
  		}
    	cudaMalloc(&endpoints1_x, sizeof(int)*BRIEF_SIZE);
    	cudaMalloc(&endpoints1_y, sizeof(int)*BRIEF_SIZE);
    	cudaMalloc(&endpoints2_x, sizeof(int)*BRIEF_SIZE);
    	cudaMalloc(&endpoints2_y, sizeof(int)*BRIEF_SIZE);
    	cudaMemcpy(endpoints1_x, vecpoints1_x.data(), sizeof(int)*BRIEF_SIZE, cudaMemcpyHostToDevice);
    	cudaMemcpy(endpoints1_y, vecpoints1_y.data(), sizeof(int)*BRIEF_SIZE, cudaMemcpyHostToDevice);
    	cudaMemcpy(endpoints2_x, vecpoints2_x.data(), sizeof(int)*BRIEF_SIZE, cudaMemcpyHostToDevice);
    	cudaMemcpy(endpoints2_y, vecpoints2_y.data(), sizeof(int)*BRIEF_SIZE, cudaMemcpyHostToDevice);
	};
	
	
	void computeBrief(float* img, int img_w, int img_h, unsigned int* locations, int numCorners, char* output);
private:
	int* endpoints1_x;
	int* endpoints1_y;
	int* endpoints2_x;
	int* endpoints2_y;
};



#endif


