// simple_camera.cpp
// MIT License
// Copyright (c) 2019 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include "grayscale.h"
#include "convolve.h"
#include "harris.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <stdio.h>

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
const int framerate = 30 ;


// Kernel for grayscale
const dim3 grayBlockSize(32);
const dim3 grayGridSize((capture_height*capture_width+grayBlockSize.x - 1)/grayBlockSize.x);
    
    
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
    float* grayBuf;
    cudaMalloc(&grayBuf,sizeof(float)*capture_height*capture_width);
    
    float* floatBuf2;
    cudaMalloc(&floatBuf2, sizeof(float)*capture_height*capture_width);
    
    float* floatBuf3;
    cudaMalloc(&floatBuf3, sizeof(float)*capture_height*capture_width);
    
    char* charBuf;
    cudaMalloc(&charBuf, sizeof(char)*capture_height*capture_width);
    
    //instantiate GPU objects
    //float jasmineBlur[9] = {0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.1};
    //Convolve<3, 3> blur(jasmineBlur);
    
    float sobelX[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
    Convolve<3, 3> gradX(sobelX);
    
    float sobelY[9] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    Convolve<3, 3> gradY(sobelY);
    double gradAvg=0;
    int n=0;
    while(true)
    {
    	//auto start=tic();
    	if (!cap.read(img)) {
			std::cout<<"Capture read error"<<std::endl;
			break;
		}
		//printf("Image size: %d,%d\n",img.cols,img.rows);
		
		// Begin grayscale
		auto t=tic();
		cv::Mat gray;
		img.convertTo(gray,CV_32F,1/255.0);
		// End grayscale
		
		// Begin convolution
		t=tic();
		cudaMemcpy(grayBuf, gray.ptr(), sizeof(float)*gray.rows*gray.cols,cudaMemcpyHostToDevice);
		//toc("opencv->cuda",t);
		gradX.doConvolve(grayBuf, img.cols, img.rows, floatBuf2);
		gradY.doConvolve(grayBuf, img.cols, img.rows, floatBuf3);
		cudaCheckError(cudaDeviceSynchronize());
		//gradAvg+=toc("gradients",t);
		n++;
		
		// Do Harris Corners
		harris(floatBuf2, floatBuf3, img.cols, img.rows, grayBuf, charBuf);
		cudaCheckError(cudaDeviceSynchronize());
		gradAvg+=toc("harris",t);
		// Copy to display
		//cv::Mat matt(img.rows,img.cols,CV_32F);
		//cudaMemcpy(matt.ptr(), grayBuf, sizeof(float)*gray.rows*gray.cols, 
		//		cudaMemcpyDeviceToHost);
		cv::Mat matt(img.rows,img.cols,CV_8UC1);
		cudaMemcpy(matt.ptr(), charBuf, sizeof(char)*img.rows*img.cols, cudaMemcpyDeviceToHost);
		//toc("cuda->opencv",t);
		cv::imshow("CSI Camera",matt);
		int keycode = cv::waitKey(1) & 0xff ; 
		if (keycode == 27) break ;
		//toc("total",start);
    }
    printf("Average compute time: %f\n",gradAvg/n);
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
