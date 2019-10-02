#include <iostream>
using namespace std;


__global__ void VecAdd(float *A, float *B, float *C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
    float *A, *B, *C, *devA, *devB, *devC;
    int n = 1024*1024*32;
    A = (float*)malloc(n * sizeof(float));
    B = (float*)malloc(n * sizeof(float));
    C = (float*)malloc(n * sizeof(float));

    cudaMalloc(&devA, n * sizeof(float));
    cudaMalloc(&devB, n * sizeof(float));
    cudaMalloc(&devC, n * sizeof(float));

    for(int i = 0;i < n;i ++){
        A[i] = float(rand()) + float(rand()%100);
        B[i] = float(rand()%20) + float(rand() + 32) + 93.23;
    }

    cudaMemcpy(devA, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n * sizeof(float), cudaMemcpyHostToDevice);

    VecAdd<<<1, 1024>>>(devA, devB, devC);

    cudaMemcpy(C, devC, n * sizeof(float), cudaMemcpyDeviceToHost);

    cout << endl << C[100];
}