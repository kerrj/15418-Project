#ifndef VOODOO_H
#define VOODOO_H
struct FeatureDist{
	int distance;
	unsigned short rowIndex;
	unsigned short colIndex;
};
void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, FeatureDist *output);
#endif
