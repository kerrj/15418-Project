#ifndef VOODOO_H
#define VOODOO_H
const int NUM_PREFS = 10;
struct FeatureDist{
	int distance;
	unsigned short f1Index;//f stands for feature
	unsigned short f2Index;
	
	bool operator<(const FeatureDist other){
		return distance<other.distance;
	}
};
void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, FeatureDist *output, FeatureDist* outputTranspose);


#endif
