#include <iostream>
#include <cstdlib.h>


__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];

	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index+offset]);
		offset += stride;
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x+i]);
			__syncthreads();
			i /= 2;
		}

		if(threadIdx.x == 0){
			while(atomicCas(mutex, 0, 1) != 0);
			*max = fmaxf(*max, cache[0]);
			atomicExch(mutex, 0);
		}
	}
}

int main()
{
	unsigned int N = 1024*1024;
	float *h_array;
	float *d_array;
	float *h_max;
	float *d_max;

	h_array = (float*)malloc(N*sizeof(float));
	h_max = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_array, N*sizeof(float));
	cudaMalloc((void**)&d_max, sizeof(float));

	free(h_array);
	free(h_max);
	cudaFree(d_array);
	cudaFree(d_max);
}
