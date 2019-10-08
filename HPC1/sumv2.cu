#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;
using namespace std::chrono;

__global__ void reduce(float *g_idata, float *g_odata){
    extern __shared__ float sdata[];

    //each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = 1;s < blockDim.x; s *= 2){
        if(tid % (2 * s) == 0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void sum_CPU(float *host_input, float *host_output, unsigned int size){
    double result = 0.0f;
    auto start = high_resolution_clock::now();
    for(int i = 0;i < size;i ++){
        result += host_input[i];
    }
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();
    cout << endl << "Time required for CPU : " << time_req << " microseconds "<< endl;
    cout << endl << " Sum from CPU : " << result << endl;
}

void compute_sum_cpu(int *cpu_input, int *cpu_output, unsigned int n){
    
    for(unsigned int i = 0;i < n;i ++){
        cpu_output[0] += cpu_input[i];
    }
    
}

int main(){
    
    int maxThreads = 1024;
    
    float *host_input, *host_output, *device_input, *device_output;
    float *cpu_input;
    float *cpu_output;

    int n = 2 << 29;
    size_t size = n * sizeof(float);

    //CPU sum
    cpu_input = (float *)malloc(sizeof(float));
    cpu_output = (float *)malloc(sizeof(float));
    cpu_output[0] = 0.0f;

    for(unsigned int i = 0;i < n;i ++){
        cpu_input[i] = 1.0f;
    }

    sum_CPU(cpu_input, cpu_output, n);

    host_input = (float *)malloc(size);
    for(int i = 0;i < n;i ++){
        host_input[i] = 1;
    }
    
    int blocks = n / maxThreads;
    host_output = (float *)malloc(blocks * sizeof(float));

    const dim3 block_size(maxThreads, 1, 1);
    const dim3 grid_size(blocks, 1, 1);
    
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, blocks * sizeof(float));

    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    reduce<<<grid_size, block_size, maxThreads * sizeof(float)>>>(device_input, device_output);

    cudaMemcpy(host_output, device_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 1;i < blocks; i++){
        host_output[0] += host_output[i];
    }

    cout << endl << " Sum from GPU : " << *host_output << endl;
}