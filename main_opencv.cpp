// simple_camera.cpp
// MIT License
// Copyright (c) 2019 JetsonHacks
// See LICENSE for OpenCV license and additional information
// Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
// NVIDIA Jetson Nano Developer Kit using OpenCV
// Drivers for the camera and OpenCV are included in the base image

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <algorithm>
#include <ctype.h>

// Timing 
std::chrono::time_point<std::chrono::system_clock> tic(){
	return std::chrono::system_clock::now();
}
double toc(std::string msg,std::chrono::time_point<std::chrono::system_clock> & t){
	std::chrono::duration<double> elapsed=std::chrono::system_clock::now()-t;
	printf("Time for %s: %f\n",msg.c_str(),elapsed.count());
	t = tic();
	return elapsed.count();
}

std::string gstreamer_pipeline (int capture_width, int capture_height, int framerate) {
     return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width="+std::to_string(capture_width)+",height="+std::to_string(capture_height)+",format=NV12,framerate="+std::to_string(framerate)+"/1 ! nvvidconv ! video/x-raw,format=GRAY8 ! appsink";
}

// Camera Params
const int capture_width = 1920;
const int capture_height = 1080 ;
const int framerate = 20 ;
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
    
  	//Timing variables
  	std::vector<double> timeTable(10);
  	int numRounds = 0;
  	std::vector<cv::Point2f> points;
    while(true)
    {	
    	if (!cap.read(img)) {
			std::cout<<"Capture read error"<<std::endl;
			break;
		}
		numRounds++;
		printf("\n");
		// Begin grayscale
		auto t=tic();
		auto start=tic();
		cv::Mat gray;
		img.convertTo(gray,CV_32F,1/255.0);
		// End grayscale
		
		// Do gaussian blur
		cv::Mat blur;
		cv::GaussianBlur(gray, blur, cv::Size(3, 3), 1, 1, cv::BORDER_DEFAULT);
		timeTable[0] += toc("blur",t);
		// Do Harris Corners
		goodFeaturesToTrack(blur, points, NUM_CORNERS, 0.1, 0, cv::noArray(), 9, true, 0.05);
		
		timeTable[1] += toc("harris",t);
		
		// Copy to display
		for(size_t i = 0; i < points.size(); i++) {
			cv::Point start(points[i].x,points[i].y);
			cv::drawMarker(gray, start, 1.0f);
		}
		
		timeTable[9] +=toc("draw",t);
		cv::imshow("CSI Camera", gray);
		int keycode = cv::waitKey(1) & 0xff ; 
		if (keycode == 27) break ;
    }

    
    // Print timing
    printf("Convolve, Harris, Scan, Collapse, Select Corners, Brief, Dist Matrix, Partial Sort Prefs, Marriage, Draw\n");
    for(double t : timeTable) printf("%f ", t/numRounds);
    printf("\n");
    
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}
