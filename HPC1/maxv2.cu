#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define max(a, b) a>b?a:b

__global__ void compute_max_gpu(float *device_input, float *device_output){
    extern __shared__ float sm[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sm[tid] = device_input[i];
    __syncthreads();

    for(int s = 1;s < blockDim.x; s*= 2){
        if(tid % (2 * s) == 0){
            sm[tid] = max(sm[tid], sm[tid+s]);
        }
        __syncthreads();
    }

    if(tid == 0) device_output[blockIdx.x] = sm[0];
}

void compute_max_cpu(float *cpu_input, float *cpu_output, unsigned int n){
    cpu_output[0] = FLT_MIN;
    
    auto start = high_resolution_clock::now();
    for(int i = 0;i < n;i ++){
        cpu_output[0] = max(cpu_output[0], cpu_input[i]);
    }
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();
    cout << endl << " Maximum from CPU is : " << cpu_output[0] << endl;
    cout << endl << " Time required for CPU is : " << time_req << " microseconds" << endl;
}


int main(){

    float *cpu_input, *cpu_output;
    float *device_input, *device_output, *transfer_output;

    unsigned int n = 2 << 29;
    size_t size = n * sizeof(float);
    
    unsigned int numThreadsPerBlock = 1024;
    unsigned int blocksPerGrid = int(n / numThreadsPerBlock);

    // Allocate memories
    cpu_input = (float *)malloc(size);
    cpu_output = (float *)malloc(sizeof(float));
    transfer_output = (float *)malloc(blocksPerGrid * sizeof(float));
    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, blocksPerGrid * sizeof(float));

    //Fill in the arrays
    for(unsigned int i = 0;i < n;i ++){

        if(i == 23842){
            cpu_input[i] = 101.0; // maximum for testing
        }
        else{
            cpu_input[i] = 99.0;
        }
    }
    for(int i = 0;i < n;i ++){

    }

    cudaMemcpy(device_input, cpu_input, size, cudaMemcpyHostToDevice);

    //Execute CPU code for maximum
    compute_max_cpu(cpu_input, cpu_output, n);

    //Execute GPU code for maximum
    dim3 grid_size(blocksPerGrid, 1, 1);
    dim3 block_size(numThreadsPerBlock, 1, 1);

    auto start = high_resolution_clock::now();
    compute_max_gpu<<<grid_size, block_size, numThreadsPerBlock * sizeof(float)>>>(device_input, device_output);
    cudaMemcpy(transfer_output, device_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    // Compute the maximum from the final array
    float max = FLT_MIN;
    for(int i = 0;i < blocksPerGrid;i ++){
        max = (max > transfer_output[i])?max:transfer_output[i];
    }
    auto stop = high_resolution_clock::now();
    auto time_req = duration_cast<microseconds>(stop - start).count();
    cout << endl << " Maximum from GPU is : " << max << endl;
    cout << endl << " Time required for GPU is : " << time_req << " microseconds" << endl;
    
}