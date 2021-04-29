#ifndef VOODOO_H
#define VOODOO_H
const int NUM_PREFS = 10;
struct FeatureDist{
	int distance;
	short f1Index;//f stands for feature
	short f2Index;
	
	bool operator<(const FeatureDist other){
		return distance<other.distance;
	}
};
void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, FeatureDist *output, FeatureDist* outputTranspose);

void galeShapley(FeatureDist *pref1, FeatureDist *pref2, int size1, int size2);

#endif
