#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;
using namespace std::chrono;

__global__ void reduce(int *g_idata, int *g_odata){
    extern __shared__ int sdata[];

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

void sum_CPU(int *host_input, int *host_output, unsigned int size){
    host_output[0] = 0;
    auto start = high_resolution_clock::now();
    for(int i = 0;i < size;i ++){
        host_output[0] += host_input[i];
    }
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();
    cout << endl << "Time required for CPU : " << time_req << " microseconds "<< endl;
    cout << endl << " Sum from CPU : " << host_output[0] << endl;
}

void compute_sum_cpu(int *cpu_input, int *cpu_output, unsigned int n){
    
    for(unsigned int i = 0;i < n;i ++){
        cpu_output[0] += cpu_input[i];
    }
    
}

int main(){
    
    int maxThreads = 1024;
    
    int *host_input, *host_output, *device_input, *device_output;
    int *cpu_input, *cpu_output;

    int n = 2 << 21;
    size_t size = n * sizeof(int);

    //CPU sum
    cpu_input = (int *)malloc(size);
    cpu_output = (int *)malloc(sizeof(int));
    cpu_output[0] = 0;

    for(unsigned int i = 0;i < n;i ++){
        cpu_input[i] = 1;
    }

    sum_CPU(cpu_input, cpu_output, n);

    host_input = (int *)malloc(size);
    for(int i = 0;i < n;i ++){
        host_input[i] = 1;
    }
    
    int blocks = n / maxThreads;
    host_output = (int *)malloc(blocks * sizeof(int));

    const dim3 block_size(maxThreads, 1, 1);
    const dim3 grid_size(blocks, 1, 1);
    
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, blocks * sizeof(int));

    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

    reduce<<<grid_size, block_size, maxThreads * sizeof(int)>>>(device_input, device_output);

    cudaMemcpy(host_output, device_output, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 1;i < blocks; i++){
        host_output[0] += host_output[i];
    }

    cout << endl << " Sum from GPU : " << *host_output << endl;
}