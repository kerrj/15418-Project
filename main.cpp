// simple_camera.cpp
// MIT License
// Copyright (c) 2019 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "grayscale.h"
#include "convolve.h"
#include "harris.h"
#include "brief.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <stdio.h>
#include "matching.h"
#include <algorithm>

// Timing 
std::chrono::time_point<std::chrono::system_clock> tic(){
	return std::chrono::system_clock::now();
}
double toc(std::string msg,std::chrono::time_point<std::chrono::system_clock> t){
	std::chrono::duration<double> elapsed=std::chrono::system_clock::now()-t;
	printf("Time for %s: %f\n",msg.c_str(),elapsed.count());
	return elapsed.count();
}
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char*file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",cudaGetErrorString(code), file, line);
		if (abort) exit(code);
    }
}

std::string gstreamer_pipeline (int capture_width, int capture_height, int framerate) {
     return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width="+std::to_string(capture_width)+",height="+std::to_string(capture_height)+",format=NV12,framerate="+std::to_string(framerate)+"/1 ! nvvidconv ! video/x-raw,format=GRAY8 ! appsink";
}

// Camera Params
const int capture_width = 1280;
const int capture_height = 720 ;
const int framerate = 20 ;
const short NUM_CORNERS = 300;//TODO play with this 


// Kernel for grayscale
const dim3 grayBlockSize(32);
const dim3 grayGridSize((capture_height*capture_width+grayBlockSize.x - 1)/grayBlockSize.x);
 
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
    
int main()
{
	// Open camera stream
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
		std::cout<<"Failed to open camera."<<std::endl;
		return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat img;

    std::cout << "Hit ESC to exit" << "\n" ;
    // End open camera stream
    
    // Malloc stuff for kernels
    float* floatBuf1;
    cudaMalloc(&floatBuf1,sizeof(float)*capture_height*capture_width);
    
    float* blurBuf;
    cudaMalloc(&blurBuf,sizeof(float)*capture_height*capture_width);
    
    float* floatBuf2;
    cudaMalloc(&floatBuf2, sizeof(float)*capture_height*capture_width);
    
    float* floatBuf3;
    cudaMalloc(&floatBuf3, sizeof(float)*capture_height*capture_width);
    
    char* featureBuf1;
    cudaMalloc(&featureBuf1, sizeof(char)*NUM_CORNERS*(CHARS_PER_BRIEF));
    
    char* featureBuf2;
    cudaMalloc(&featureBuf2, sizeof(char)*NUM_CORNERS*(CHARS_PER_BRIEF));
    
    unsigned short* shortBuf1;
    cudaMalloc(&shortBuf1, sizeof(short)*nextPow2(capture_height*capture_width));
    
    unsigned int* intBuf1;
    cudaMalloc(&intBuf1, sizeof(int)*capture_height*capture_width);
    
    FeatureDist* prefBuf1;
    cudaMalloc(&prefBuf1, sizeof(FeatureDist)*NUM_CORNERS*NUM_PREFS);
    
    FeatureDist* prefBuf2;
    cudaMalloc(&prefBuf2, sizeof(FeatureDist)*NUM_CORNERS*NUM_PREFS);
    
    FeatureDist *distMatrixBuf;
    cudaMalloc(&distMatrixBuf,sizeof(FeatureDist)*NUM_CORNERS*NUM_CORNERS);
    
    FeatureDist *distMatrixBufTranspose;
    cudaMalloc(&distMatrixBufTranspose,sizeof(FeatureDist)*NUM_CORNERS*NUM_CORNERS);
    
    //instantiate GPU objects
    const int blurSize = 3;
    cv::Mat blurKern = cv::getGaussianKernel(blurSize,2,CV_32F);
    Convolve<1, blurSize> blurX((float*)blurKern.ptr());
    Convolve<blurSize, 1> blurY((float*)blurKern.ptr());
    
    float sobelX[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
    Convolve<3, 3> gradX(sobelX);
    
    float sobelY[9] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    Convolve<3, 3> gradY(sobelY);
    double gradAvg=0;
    int n=0;
    
    Brief brief;
  	int frameNum = 0;
  	
  	int prevFeatureSize;
  	
    while(true)
    {
    	if (!cap.read(img)) {
			std::cout<<"Capture read error"<<std::endl;
			break;
		}
		// Begin grayscale
		auto t=tic();
		cv::Mat gray;
		img.convertTo(gray,CV_32F,1/255.0);
		// End grayscale
		//copy memory to cuda land
		cudaMemcpy(floatBuf1, gray.ptr(), sizeof(float)*gray.rows*gray.cols,cudaMemcpyHostToDevice);
		
		//blur the image
		blurX.doConvolve(floatBuf1,img.cols,img.rows,floatBuf2);
		blurY.doConvolve(floatBuf2,img.cols,img.rows,blurBuf);
		// Begin convolution
		t=tic();
		gradX.doConvolve(blurBuf, img.cols, img.rows, floatBuf2);
		gradY.doConvolve(blurBuf, img.cols, img.rows, floatBuf3);
		n++;
		cudaDeviceSynchronize();
		toc("convolve",t);
		// Do Harris Corners
		harris(floatBuf2, floatBuf3, img.cols, img.rows, floatBuf1, shortBuf1);
		// Do Scan. Input: activations. Output: int array with scanned count.
		scan(shortBuf1, img.cols * img.rows);
		short numCorners;
		cudaMemcpy(&numCorners,&shortBuf1[img.cols*img.rows-1],sizeof(short),cudaMemcpyDeviceToHost);
		numCorners = std::min(NUM_CORNERS,numCorners);
		if(numCorners < NUM_PREFS){
			frameNum=0;
			printf("No corners found\n");
			continue;
		}
		// Collapse to # of corners
		collapse(shortBuf1, floatBuf1, img.cols * img.rows, intBuf1, floatBuf2);
		//TODO sort corners to pick the highest N
		//find image features
		char* prevFeatures, *curFeatures;
		cudaDeviceSynchronize();
		toc("corners",t);
		if(frameNum & 1) {
			brief.computeBrief(blurBuf,img.cols, img.rows, intBuf1,numCorners,featureBuf1);
			prevFeatures=featureBuf2;
			curFeatures=featureBuf1;
		} else {
			brief.computeBrief(blurBuf,img.cols, img.rows, intBuf1,numCorners,featureBuf2);
			prevFeatures=featureBuf1;
			curFeatures=featureBuf2;
		}
		cudaDeviceSynchronize();
		toc("brief",t);
		
		if(frameNum++ == 0){
			prevFeatureSize=numCorners;
			continue;
		}
		// Create distance matrix (GPU)
		makeDistMatrix(prevFeatures,curFeatures,prevFeatureSize,numCorners,
						distMatrixBuf,distMatrixBufTranspose);
		cudaDeviceSynchronize();
		toc("dist matrix",t);
		/*
		Plan
		1. Store featureBuf in alternating buffers
		2. Create distance matrix between features using hamming distance
		3. Find top ten on CPU using openMP/partial sort
		4. Marriage-sort like algorithm for bipartite matching TODO
		*/
		cudaDeviceSynchronize();
		std::vector<FeatureDist> distHost(prevFeatureSize * numCorners);
		cudaMemcpy(distHost.data(), distMatrixBuf, sizeof(FeatureDist)*prevFeatureSize*numCorners, cudaMemcpyDeviceToHost);
		//sort each row and store result in prefs1
		toc("before rows",t);
		std::vector<FeatureDist> pref1Vec(NUM_PREFS*numCorners);
		#pragma omp parallel for
		for(int r=0;r<numCorners;r++){
			auto start = distHost.begin() + r*prevFeatureSize;
			auto end = start + prevFeatureSize;
			std::partial_sort(start,start+NUM_PREFS,end);
			std::copy(start,start+NUM_PREFS,&pref1Vec[r*NUM_PREFS]);
		}
		toc("after rows",t);
		//copy the result into the prefBuf1
		cudaMemcpy(prefBuf1,pref1Vec.data(),pref1Vec.size()*sizeof(FeatureDist),
						cudaMemcpyHostToDevice);
		toc("after copy",t);
		//sort each col and store result in presf2
		cudaMemcpy(distHost.data(), distMatrixBufTranspose, sizeof(FeatureDist)*prevFeatureSize*numCorners, cudaMemcpyDeviceToHost);
		//WARNING this overwrites distHost
		std::vector<FeatureDist> pref2Vec(NUM_PREFS*prevFeatureSize);
		#pragma omp parallel for
		for(int c=0;c<prevFeatureSize;c++){
			auto start = distHost.begin() + c*numCorners;
			auto end = start + numCorners;
			std::partial_sort(start,start+NUM_PREFS,end);
			std::copy(start,start+NUM_PREFS,&pref2Vec[c*NUM_PREFS]);
		}
		cudaMemcpy(prefBuf2,pref2Vec.data(),pref2Vec.size()*sizeof(FeatureDist),
						cudaMemcpyHostToDevice);
		toc("partial sort preferences",t);
		
		//visualize the dists in an image
		cv::Mat diffMat(prevFeatureSize, numCorners, CV_16U);
		for(int i = 0; i < prevFeatureSize*numCorners; i++) {
			diffMat.at<short>(i) = distHost[i].distance * 100;
		}
		
		prevFeatureSize=numCorners;
		// Copy to display
		std::vector<unsigned int> cornerIds(numCorners);
		cudaMemcpy(cornerIds.data(),intBuf1,sizeof(int)*numCorners,cudaMemcpyDeviceToHost);
		for(int i=0;i<numCorners;i++){
			unsigned int index = cornerIds[i];
			unsigned int rId = index / img.cols;
			unsigned int cId = index % img.cols;
			cv::Point p(cId,rId);
			cv::drawMarker(gray,p,255);
		}
		toc("draw",t);
		cv::imshow("CSI Camera", diffMat);
		int keycode = cv::waitKey(1) & 0xff ; 
		if (keycode == 27) break ;
    }
    printf("Average compute time: %f\n",gradAvg/n);
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
