#include "brief.h"

__global__ void _brief(float* img, int img_w, int img_h,unsigned int* locations, int numCorners, int* endpoints1_x, int* endpoints1_y, int* endpoints2_x, int* endpoints2_y, char* output) {
	// Get pixel location
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int location = locations[idx];
	const int x = location % img_w;
	const int y = location / img_w;
	
	if(x - (WINDOW/2) < 0 || x + (WINDOW/2) >= img_w) return;
	if(y - (WINDOW/2) < 0 || y + (WINDOW/2) >= img_h) return;

	// outerloop will create each char and write to output buf
	for(int charNum = 0; charNum < CHARS_PER_BRIEF; charNum++) {
		char bitVec = 0;
		// inner loop will do 8 features and add to bit vector
		for(int i = 0; i < 8; i++) {
			const int featureNum = charNum * 8 + i;
			const int point1_x = x + endpoints1_x[featureNum];
			const int point1_y = y + endpoints1_y[featureNum];
			const int point2_x = x + endpoints2_x[featureNum];
			const int point2_y = y + endpoints2_y[featureNum];
			
			// Convert x, y back into indices and grab values from img
			const float pixel1_value = img[point1_x + point1_y * img_w];
			const float pixel2_value = img[point2_x + point2_y * img_w];
			
			// Compare values
			if(pixel1_value > pixel2_value) {
				bitVec += 1 << i;
			} 
		}
		// Write bit vector to output
		output[(CHARS_PER_BRIEF) * idx + charNum] = bitVec;
	}
	
}
void Brief::computeBrief(float* img, int img_w, int img_h, unsigned int* locations, int numCorners, char* output){
		const int threadsPerBlock = 128;
		const int blocks = (numCorners + threadsPerBlock - 1) / threadsPerBlock;
		_brief<<<blocks, threadsPerBlock>>>(img, img_w, img_h, locations, numCorners, endpoints1_x, endpoints1_y, endpoints2_x,endpoints2_y, output);
}
