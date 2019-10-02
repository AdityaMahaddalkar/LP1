#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

__global__ void vecAddGPU(double *a, double *b, double *c, double n){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < n){
        c[id] = a[id] + b[id];
    }
}

void vecAddCPU(double *a, double *b, double *c, double n){

    for(int i = 0;i < n;i ++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    double n = 100000000;

    double *a, *b, *c, *dev_a, *dev_b, *dev_c;

    //Allocate memories
    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    c = (double*)malloc(n * sizeof(double));
    cudaMalloc(&dev_a, n * sizeof(double));
    cudaMalloc(&dev_b, n * sizeof(double));
    cudaMalloc(&dev_c, n * sizeof(double));

    // Get random double numbers
    for(int i = 0;i < n;i ++){
        a[i] = double(rand());
        b[i] = double(rand());
    }

    // Time CPU execution
    auto startCPU = high_resolution_clock::now();
    vecAddCPU(a, b, c, n);
    auto stopCPU = high_resolution_clock::now();

    //Move a and b to device
    cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Time GPU execution
    auto startGPU = high_resolution_clock::now();
    vecAddGPU<<<1, 256>>>(dev_a, dev_b, dev_c, n);
    auto stopGPU = high_resolution_clock::now();

    //Compare execution times
    cout << endl;
    cout << "====Execution times===" << endl;
    cout << " CPU (in microseconds): " << duration_cast<microseconds>(stopCPU - startCPU).count() << endl;
    cout << " GPU (in microseconds): " << duration_cast<microseconds>(stopGPU - startGPU).count() << endl;
    cout << endl;

}