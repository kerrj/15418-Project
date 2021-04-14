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
const int framerate = 20 ;


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
    
    float* floatBuf2;
    cudaMalloc(&floatBuf2, sizeof(float)*capture_height*capture_width);
    
    float* floatBuf3;
    cudaMalloc(&floatBuf3, sizeof(float)*capture_height*capture_width);
    
    char* charBuf;
    cudaMalloc(&charBuf, sizeof(char)*capture_height*capture_width);
    
    unsigned short* shortBuf1;
    cudaMalloc(&shortBuf1, sizeof(short)*nextPow2(capture_height*capture_width));
    
    unsigned short* shortBuf2;
    cudaMalloc(&shortBuf2, sizeof(short)*capture_height*capture_width);
    
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
		//copy memory to cuda land
		cudaMemcpy(floatBuf1, gray.ptr(), sizeof(float)*gray.rows*gray.cols,cudaMemcpyHostToDevice);
		
		//blur the image
		blurX.doConvolve(floatBuf1,img.cols,img.rows,floatBuf2);
		blurY.doConvolve(floatBuf2,img.cols,img.rows,floatBuf1);
		// Begin convolution
		t=tic();
		//toc("opencv->cuda",t);
		gradX.doConvolve(floatBuf1, img.cols, img.rows, floatBuf2);
		gradY.doConvolve(floatBuf1, img.cols, img.rows, floatBuf3);
		//cudaCheckError(cudaDeviceSynchronize());
		//gradAvg+=toc("gradients",t);
		n++;
		
		// Do Harris Corners
		harris(floatBuf2, floatBuf3, img.cols, img.rows, floatBuf1, shortBuf1);
		
		// Do Scan. Input: activations. Output: int array with scanned count.
		scan(shortBuf1, img.cols * img.rows);
		// Collapse to # of corners. Input: scan results + activations, size img. Output: 1 int array for locations, 1 float array for activations, size # of corners
		//collapse(shortBuf1, floatBuf1, img.cols * img.rows, shortBuf2, floatBuf2);
		
		
		// cudaCheckError(cudaDeviceSynchronize());
		gradAvg+=toc("harris",t);
		// Copy to display
		//cv::Mat matt(img.rows,img.cols,CV_32F);
		//cudaMemcpy(matt.ptr(), floatBuf1, sizeof(float)*gray.rows*gray.cols, cudaMemcpyDeviceToHost);
		cv::Mat matt(img.rows,img.cols,CV_16UC1);
		cudaMemcpy(matt.ptr(), shortBuf1, sizeof(short)*img.rows*img.cols, cudaMemcpyDeviceToHost);
		//toc("cuda->opencv",t);
		cv::imshow("CSI Camera", (2 << 3) * matt);
		int keycode = cv::waitKey(1) & 0xff ; 
		if (keycode == 27) break ;
		//toc("total",start);
    }
    printf("Average compute time: %f\n",gradAvg/n);
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
