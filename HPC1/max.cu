#include <iostream>
#include <chrono>
#include <cstdlib>
#include <limits>

using namespace std;
using namespace std::chrono;

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n){

    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache[256];

    float temp = 10000000000.99f;
    while(index + offset < n){
        temp = fminf(temp, array[index + offset]);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    //reduction
    unsigned int i = blockDim.x / 2;
    while(i != 0){
        if(threadIdx.x > i){
            cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0){
        while(atomicCAS(mutex, 0, 1) != 0); //lock
        *max = fminf(*max, cache[0]);
        atomicExch(mutex, 0);
    }
}

void find_maximum_CPU(float *a, int n){
    float max = 100000000000.99f;
    for(int i = 0;i < n;i ++){
        if(a[i] < max){
            max = a[i];
        }
    }
    cout << "\nThe max number from CPU is: " << max << "\n";
}

int main(){

    float *a, *dev_a, *d_max, *h_max;
    int *d_mutex;
    int n = 1024 * 1024 * 20;

    //Allocate memory to array a and h_max
    a = (float *)malloc(n * sizeof(float));
    h_max = (float*)malloc(n * sizeof(float));

    for(int i = 0;i < n;i ++){
        a[i] = float(rand()) + 100.0f;
    }

    //Find max w/ CPU
    auto startCPU = high_resolution_clock::now();
    find_maximum_CPU(a, n);
    auto stopCPU = high_resolution_clock::now();
    cout << "\nTime elapsed for CPU: " << duration_cast<microseconds>(stopCPU - startCPU).count() << " microseconds\n"; 

    //Allocate CUDA memory to dev_a, d_max, d_mutex
    cudaMalloc(&dev_a, n * sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_mutex, sizeof(int));
    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_mutex, 0, sizeof(float));

    //Move array a to dev_a
    cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);

    //Apply kernel on dev_a
    dim3 gridSize = 256;
    dim3 blockSize = 256;
    auto startGPU = high_resolution_clock::now();
    find_maximum_kernel<<< gridSize, blockSize >>> (dev_a, d_max, d_mutex, n);
    auto stopGPU = high_resolution_clock::now();
    


    //Copy from device to host
    cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    //Report result from CUDA
    cout << "\nMax Number from GPU is: " << *h_max << "\n";
    cout << "\nTime elapsed for GPU: " << duration_cast<microseconds>(stopGPU - startGPU).count() << " microseconds\n";

    cudaFree(dev_a);

}