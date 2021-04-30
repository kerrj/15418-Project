#ifndef VOODOO_H
#define VOODOO_H
const int NUM_PREFS = 3;//has to be at least 3 otherwise flags die
const int MATCH_THRESHOLD = 500;
struct FeatureDist{
	int featureDistance;
	int flag;
	short f1Index;//f stands for feature
	short f2Index;
	
	bool operator<(const FeatureDist other){
		return featureDistance<other.featureDistance;
	}
};
void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, FeatureDist *output, FeatureDist* outputTranspose);

void galeShapley(FeatureDist *pref1, FeatureDist *pref2, int size1, int size2);

#endif
