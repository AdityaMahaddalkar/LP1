#include <iostream>
#include <cmath>
#include <chrono>
#define N 1000
using namespace std;
using namespace std::chrono;

__global__ void ArraySum(float *array, float *sum){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N)
        atomicAdd(sum, array[index]);

}

void findSum(float *array, float *sum){

    auto start = high_resolution_clock::now();
    for(int i = 0;i < N;i ++){
        *sum += array[i];
    }
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();
    cout << endl << "Sum from CPU is: " << *sum << endl;
    cout << endl << "Time required for CPU: " << time_req << endl;
}

int main(){

    float *hostInput, *deviceInput, *sumCPU, *sumGPU, *sumGPU2CPU;

    hostInput = (float*)malloc(N * sizeof(float));
    sumCPU = (float*)malloc(sizeof(float));
    sumGPU2CPU = (float*)malloc(sizeof(float));
    *sumCPU = 0;
    for(int i = 0;i < N;i ++){
        hostInput[i] = 1.0f;
    }

    cudaMalloc(&deviceInput, N * sizeof(float));
    cudaMalloc(&sumGPU, sizeof(float));

    cudaMemcpy(deviceInput, hostInput, N * sizeof(float), cudaMemcpyHostToDevice);

    findSum(hostInput, sumCPU);

    dim3 threadsPerBlock(512, 1, 1);
    dim3 numBlocks(512, 1, 1);

    auto start = high_resolution_clock::now();
    ArraySum<<<numBlocks, threadsPerBlock>>>(deviceInput, sumGPU);
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();

    cudaMemcpy(sumGPU2CPU, sumGPU, sizeof(float), cudaMemcpyDeviceToHost);

    cout << endl << "Sum from GPU is: " << *sumGPU2CPU << endl;
    cout << endl << "Time required for GPU: " << time_req << endl;

    free(hostInput);
    free(sumCPU);
    free(sumGPU2CPU);
    cudaFree(deviceInput);
    cudaFree(sumGPU);

}