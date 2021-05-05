#include "matching.h"
#include "brief.h"

__device__ int numberOfSetBits(char c)
{
     int numBits = 0;
     for(int i = 0; i < 8; i++) {
     	numBits += ((1 << i) & c) >> i;
     }
     
     return numBits;
}

__device__ int numberOfSetBitsInt(int i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

__global__ void _makeDistMatrix(const char* featureBuf1, const char* featureBuf2, int size1, int size2,int* locs1, int* locs2, FeatureDist* output, FeatureDist* outputTranspose, int img_w, int img_h){
	// Matrix is size2 rows by size1 columns. 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x>=size1 || y>=size2)return;
	// Get the feature from location x and y
	int diff = 0;
	for(int i=0;i < INTS_PER_BRIEF; i++){
		const int f1 = ((int*)featureBuf1)[INTS_PER_BRIEF * x + i];
		const int f2 = ((int*)featureBuf2)[INTS_PER_BRIEF * y + i];
		diff += numberOfSetBitsInt(f1 ^ f2);
	}
	const int yloc1 = locs1[x] / img_w;
	const int xloc1 = locs1[x] - yloc1*img_w;
	const int yloc2 = locs2[y] / img_w;
	const int xloc2 = locs2[y] - yloc2*img_w;
	
	diff += 2*(std::abs(xloc1-xloc2)+std::abs(yloc1-yloc2));
	output[x + size1 * y].featureDistance = diff;
	output[x + size1 * y].f2Index = y;
	output[x + size1 * y].f1Index = x;
	
	outputTranspose[y + size2 * x].featureDistance = diff;
	outputTranspose[y + size2 * x].f2Index = y;
	outputTranspose[y + size2 * x].f1Index = x;
}

__global__ void _propose(FeatureDist* oneRanks2, FeatureDist* twoRanks1, int size1, int size2, int roundNum) {
	// Each feature writes its preference to the first distance field
	// Check confirmation flag to second distance field 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(x >= size1) return;
	if(roundNum == 0) {
		// Set 'match' flag to false. Sets 1 if currently matched
		oneRanks2[NUM_PREFS*x + 1].flag = 0;
		//  Rank of the next pref2 to propose to
		oneRanks2[NUM_PREFS*x + 2].flag = 0; 
	} 
	// Return if matched
	if(oneRanks2[NUM_PREFS*x + 1].flag == 1) return;
	
	// Get rank of next pref2 to propose to
	int propRank = oneRanks2[NUM_PREFS*x + 2].flag++;
	// Set propose to pref2 idx
	if(oneRanks2[NUM_PREFS*x + propRank].featureDistance>MATCH_THRESHOLD)return;
	oneRanks2[NUM_PREFS*x].flag = oneRanks2[NUM_PREFS*x + propRank].f2Index;
}

__global__ void _check(FeatureDist* oneRanks2, FeatureDist* twoRanks1, int size1, int size2, int roundNum) {
	// Each feature writes its preference to the first distance field
	// Check confirmation flag to second distance field 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(x >= size2) return;
	
	if(roundNum == 0) {
		// Set rank of best match so far (default to NUM_PREFS)
		twoRanks1[NUM_PREFS*x + 1].flag = NUM_PREFS; 
	} 
	FeatureDist* pref2CurRank = &twoRanks1[NUM_PREFS*x +1]; 
	for(int i = 0; i < NUM_PREFS; i++) {
		int pref1Idx = twoRanks1[NUM_PREFS*x + i].f1Index;
		// If proposer's match is this pref2, and it is ranked higher in this preference list
		if(oneRanks2[NUM_PREFS*pref1Idx].flag == x && i < pref2CurRank->flag) {
			// Get Previous proposer's idx
			if(pref2CurRank->flag < NUM_PREFS) {
				int prevPref1Idx = twoRanks1[NUM_PREFS*x + pref2CurRank->flag].f1Index;
				// Invalidate previous pref1's match flag
				twoRanks1[NUM_PREFS*prevPref1Idx + 1].flag = 0;
			}
			
			// Set proposer 
			twoRanks1[NUM_PREFS*x].flag = pref1Idx;
			// Set match flag to index in pref-list of proposer
			pref2CurRank->flag = i;
			
			// Set pref1's match flag
			oneRanks2[NUM_PREFS*pref1Idx + 1].flag = 1;		
			return;
		}
	}
}

void galeShapleyRound(FeatureDist *pref1, FeatureDist *pref2, int size1, int size2, int roundNum) {
	// Propose kernel
	const int threadsPerBlock = 64;
	const int blocks1 = (size1 + threadsPerBlock - 1) / threadsPerBlock;
	_propose<<<blocks1, threadsPerBlock>>>(pref1, pref2, size1, size2, roundNum);

	// Check kernel
	const int blocks2 = (size2 + threadsPerBlock - 1) / threadsPerBlock;
	_check<<<blocks2, threadsPerBlock>>>(pref1, pref2, size1, size2, roundNum);
}

void galeShapley(FeatureDist *pref1, FeatureDist *pref2, int size1, int size2) {
	// TODO Might not finish everyone's list
	//printf("Note: only doing 1 iter of gale shapley\n");
	for(int i = 0; i < NUM_PREFS; i++) {
		galeShapleyRound(pref1, pref2, size1, size2, i);
	}
}

void makeDistMatrix(char* featureBuf1, char* featureBuf2, int size1, int size2, int* locs1, int* locs2, FeatureDist *output, FeatureDist* outputTranspose, int img_w, int img_h){
	
	const dim3 blockSize(32, 32);
	// Make Gridsize
	const dim3 gridDims((size1 + blockSize.x - 1) / blockSize.x,
                 (size2 + blockSize.y - 1) / blockSize.y);
                 
	_makeDistMatrix<<< gridDims, blockSize >>>(featureBuf1,featureBuf2,size1,size2,locs1,locs2,output, outputTranspose, img_w, img_h);
}
