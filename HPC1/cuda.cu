#include <iostream>
#include <cstdlib.h>
#include <cuda.h>

/* GPU code: set an array to a value */
__global__ void set_array(float *vals, float param){
	int i = threadIdx.x;
	vals[i] = i + param;
}

void main(int args, char *argv[]){
	int n = 16;
	float *vals; /* device array of n values */
	cudamalloc( (void**) &vals, n*sizeof(float)); // Allocate GPU Space

	set_array<<<1,n>>>(vals, 0.1234);

	/*Copy a few elements back to CPU for printing */
	int i = 7;
	float f = -999.0; /* CPU copy of value */
	cudaMemcpy(&f, &vals[i], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Vals[%d] = &f", i, f);

}