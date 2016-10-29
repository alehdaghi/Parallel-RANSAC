#ifndef types_h_cuda
#define types_h_cuda
#define BLOCK_SIZE 1000
#define THR_PLANE 8.5
struct xyz{
	float x,y,z;
	int i,j;
	
};

struct segment{
	xyz* points;
	int len;
};


#endif