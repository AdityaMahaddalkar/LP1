#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void scan(float *input, float *output, int len) {
    __shared__ float data[BLOCK_SIZE];

    // DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Block Number: %d\n", blockIdx.x);
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            printf("DATA[%d] = %f\n", i, data[i]);
        }
    }

}

int main(int argc, char ** argv) {
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(10, 1, 1);
    scan<<<grid,block>>>(NULL, NULL, NULL);
    cudaDeviceSynchronize();
    return 0;
}