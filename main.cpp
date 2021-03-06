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
double toc(std::string msg,std::chrono::time_point<std::chrono::system_clock> & t){
	std::chrono::duration<double> elapsed=std::chrono::system_clock::now()-t;
	//printf("Time for %s: %f\n",msg.c_str(),elapsed.count());
	t = tic();
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
const int capture_width = 640;
const int capture_height = 480 ;
const int framerate = 30 ;
const short NUM_CORNERS = 400;//TODO play with this 
const int blurSize = 3;

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
    
    cv::Mat blurKern = cv::getGaussianKernel(blurSize,1,CV_32F);
    Convolve blurX((float*)blurKern.ptr(),1,blurSize,0);
    Convolve blurY((float*)blurKern.ptr(),blurSize,1,0);
    float sobelX[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
    Convolve gradX(sobelX,3,3,1);
    
    float sobelY[9] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    Convolve gradY(sobelY,3,3,2);
    Brief brief;
  	int frameNum = 0;
  	
  	int prevFeatureSize;
  	std::vector<unsigned int> cornerLocations;
  	
  	//Timing variables
  	std::vector<double> timeTable(11);
  	int numRounds = 0;
  	
    while(true)
    {	
    	if (!cap.read(img)) {
			std::cout<<"Capture read error"<<std::endl;
			break;
		}
		numRounds++;
		//printf("\n");
		// Begin grayscale
		auto t=tic();
		auto start=tic();
		cv::Mat gray;
		img.convertTo(gray,CV_32F,1/255.0);
		// End grayscale
		//copy memory to cuda land
		auto tcpy=tic();
		cudaMemcpy(floatBuf1, gray.ptr(), sizeof(float)*gray.rows*gray.cols,cudaMemcpyHostToDevice);
		timeTable[9] += toc("cpy",tcpy);
		//blur the image
		blurX.doConvolve(floatBuf1,img.cols,img.rows,floatBuf2);
		blurY.doConvolve(floatBuf2,img.cols,img.rows,blurBuf);
		// sobel convolution
		t=tic();
		gradX.doConvolve(blurBuf, img.cols, img.rows, floatBuf2);
		gradY.doConvolve(blurBuf, img.cols, img.rows, floatBuf3);
		cudaDeviceSynchronize();
		timeTable[0] += toc("convolve",t);
		// Do Harris Corners
		harris(floatBuf2, floatBuf3, img.cols, img.rows, floatBuf1, shortBuf1);
		cudaDeviceSynchronize();
		timeTable[1] += toc("harris",t);
		// Do Scan. Input: activations. Output: int array with scanned count.
		scan(shortBuf1, img.cols * img.rows);
		cudaDeviceSynchronize();
		timeTable[2] += toc("scan",t);
		short numCorners;
		tcpy=tic();
		cudaMemcpy(&numCorners,&shortBuf1[img.cols*img.rows-1],sizeof(short),cudaMemcpyDeviceToHost);
		timeTable[9] += toc("cpy",tcpy);
		if(numCorners < NUM_PREFS){
			frameNum=0;
			printf("No corners found\n");
			continue;
		}
		// Collapse to # of corners
		collapse(shortBuf1, floatBuf1, img.cols * img.rows, intBuf1, floatBuf2);
		cudaDeviceSynchronize();
		timeTable[3] += toc("after collapse",t);
		
		std::vector<unsigned int> prevCornerLocations = cornerLocations;
		cornerLocations = selectCorners(intBuf1, floatBuf1, numCorners, 
				std::min(NUM_CORNERS,numCorners));
		cudaDeviceSynchronize();
		timeTable[4] += toc("after select corners",t);
		numCorners = std::min(NUM_CORNERS,numCorners);
		//find image features
		char* prevFeatures, *curFeatures;
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
		timeTable[5] += toc("brief",t);
		
		if(frameNum++ == 0){
			prevFeatureSize=numCorners;
			continue;
		}
		// Create distance matrix (GPU)
		//we use intbuf1 and shortbuf1 to hold the locations of 
		const int size1 = numCorners;
		const int size2 = prevFeatureSize;
		tcpy=tic();
		cudaMemcpy(intBuf1,prevCornerLocations.data(),sizeof(int)*size2,cudaMemcpyHostToDevice);
		cudaMemcpy(shortBuf1,cornerLocations.data(),sizeof(int)*size1,cudaMemcpyHostToDevice);
		timeTable[9] += toc("cpy",tcpy);
		makeDistMatrix(prevFeatures,curFeatures,prevFeatureSize,numCorners,
						(int*)intBuf1, (int*)shortBuf1,
						distMatrixBuf,distMatrixBufTranspose,img.cols,img.rows);
		cudaDeviceSynchronize();
		timeTable[6] += toc("dist matrix",t);
		
		cudaDeviceSynchronize();
		std::vector<FeatureDist> distHost(prevFeatureSize * numCorners);
		std::vector<FeatureDist> prefVec(NUM_PREFS*std::max((int)numCorners,prevFeatureSize));
		tcpy=tic();
		cudaMemcpy(distHost.data(), distMatrixBuf, sizeof(FeatureDist)*prevFeatureSize*numCorners, cudaMemcpyDeviceToHost);
		timeTable[9] += toc("cpy",tcpy);
		//sort each row and store result in prefs1
		#pragma omp parallel for
		for(int r=0;r<numCorners;r++){
			auto start = distHost.begin() + r*prevFeatureSize;
			auto end = start + prevFeatureSize;
			std::partial_sort(start,start+NUM_PREFS,end);
			std::copy(start,start+NUM_PREFS,&prefVec[r*NUM_PREFS]);
		}
		//copy the result into the prefBuf1
		tcpy=tic();
		cudaMemcpy(prefBuf1,prefVec.data(),size1*NUM_PREFS*sizeof(FeatureDist),
						cudaMemcpyHostToDevice);
		//sort each col and store result in presf2
		cudaMemcpy(distHost.data(), distMatrixBufTranspose, sizeof(FeatureDist)*prevFeatureSize*numCorners, cudaMemcpyDeviceToHost);
		timeTable[9] += toc("cpy",tcpy);
		#pragma omp parallel for
		for(int c=0;c<prevFeatureSize;c++){
			auto start = distHost.begin() + c*numCorners;
			auto end = start + numCorners;
			std::partial_sort(start,start+NUM_PREFS,end);
			std::copy(start,start+NUM_PREFS,&prefVec[c*NUM_PREFS]);
		}
		tcpy=tic();
		cudaMemcpy(prefBuf2,prefVec.data(),size2*NUM_PREFS*sizeof(FeatureDist),
						cudaMemcpyHostToDevice);
		timeTable[9] += toc("cpy",tcpy);
		timeTable[7] += toc("partial sort preferences",t);
		//prefBuf2 saves feature1's ranking of feature2's
		//prefBuf1 saves feature2's ranking of feature1's
		galeShapley(prefBuf2, prefBuf1, size2, size1);
		cudaDeviceSynchronize();
		timeTable[8] += toc("mawwiage",t);
		// Copy to display
		toc("Total compute",start);
		std::vector<FeatureDist> feature1Ranks2(NUM_PREFS*size2);
		std::vector<FeatureDist> feature2Ranks1(NUM_PREFS*size1);
		cudaMemcpy(feature1Ranks2.data(),prefBuf2,sizeof(FeatureDist)*feature1Ranks2.size(),cudaMemcpyDeviceToHost);
		cudaMemcpy(feature2Ranks1.data(),prefBuf1,sizeof(FeatureDist)*feature2Ranks1.size(),cudaMemcpyDeviceToHost);
		std::vector<unsigned int> &feature1Locations = prevCornerLocations;
		std::vector<unsigned int> &feature2Locations = cornerLocations;
		for(size_t i=0;i<size2;i++){
			if(feature1Ranks2[NUM_PREFS*i+1].flag!=1)continue;
			int imageIndex = feature1Locations.at(i);
			int rId = imageIndex / img.cols;
			int cId = imageIndex - rId*img.cols;
			int matchIn2 = feature1Ranks2[i * NUM_PREFS].flag;
			int imageIndex2 = feature2Locations.at(matchIn2);
			int rId2 = imageIndex2 / img.cols;
			int cId2 = imageIndex2 - rId2*img.cols;
			cv::Point start(cId,rId);
			cv::Point end(cId2, rId2);
			cv::arrowedLine(gray,start,end,255);
		}
		timeTable[10] +=toc("draw",t);
		cv::imshow("CSI Camera", gray);
		int keycode = cv::waitKey(1) & 0xff ; 
		if (keycode == 27) break ;
		
		prevFeatureSize=numCorners;
    }

    
    // Print timing
    printf("Convolve, Harris, Scan, Collapse, Select Corners, Brief, Dist Matrix, Partial Sort Prefs, Marriage, memcpy, Draw\n");
    double total=0;
    for(int i=0;i<timeTable.size()-2;i++){
    	total += timeTable[i]/numRounds;
    }
    for(double t : timeTable) printf("%f ", t/numRounds);
    printf("\nTotal: %f\n",total);
    printf("\n");
    
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
