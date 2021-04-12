// simple_camera.cpp
// MIT License
// Copyright (c) 2019 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include "grayscale.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <stdio.h>
std::chrono::time_point<std::chrono::system_clock> tic(){
	return std::chrono::system_clock::now();
}
void toc(std::string msg,std::chrono::time_point<std::chrono::system_clock> t){
	std::chrono::duration<double> elapsed=std::chrono::system_clock::now()-t;
	printf("%s: %f\n",msg.c_str(),elapsed.count());
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

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main()
{
    int capture_width = 1920;
    int capture_height = 1080 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 30 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat img;

    std::cout << "Hit ESC to exit" << "\n" ;
    uint8_t* imgBuf;
    cudaMalloc(&imgBuf,3*capture_height*capture_width);
    dim3 blockSize(32);
    dim3 gridSize(capture_height*capture_width/blockSize.x + 1);
    float* floatBuf;
    cudaMalloc(&floatBuf,sizeof(float)*capture_height*capture_width);
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	}
	cudaMemcpy(imgBuf, img.ptr(), 3*img.rows*img.cols,cudaMemcpyHostToDevice);
	auto t=tic();
	im2gray(imgBuf,img.rows*img.cols,floatBuf,gridSize,blockSize);
	toc("kernel",t);
	cudaCheckError(cudaDeviceSynchronize());
	cv::Mat matt(img.rows,img.cols,CV_32F);
	cudaMemcpy(matt.ptr(), floatBuf, sizeof(float)*img.rows*img.cols,cudaMemcpyDeviceToHost);
	toc("copy",t);
	cv::imshow("CSI Camera",matt);
	int keycode = cv::waitKey(30) & 0xff ; 
        if (keycode == 27) break ;
    }

    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
