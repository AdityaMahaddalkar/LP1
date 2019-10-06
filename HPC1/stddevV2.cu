#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
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

__global__ void compute_difference_between_mean_and_elements(int *difference_array, int *original_array, int mean){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    difference_array[tid] = (original_array[tid] - mean) * (original_array[tid] - mean);
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

    int n = 2 << 12;
    size_t size = n * sizeof(int);

    //CPU sum
    cpu_input = (int *)malloc(size);
    cpu_output = (int *)malloc(sizeof(int));
    cpu_output[0] = 0;

    for(unsigned int i = 0;i < n;i ++){
        cpu_input[i] = rand()%10;
    }

    sum_CPU(cpu_input, cpu_output, n);

    host_input = (int *)malloc(size);
    for(int i = 0;i < n;i ++){
        host_input[i] = cpu_input[i];
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
    
    int mean = int(host_output[0] / n);
    cout << endl << " Mean of the array : " << mean << endl;

    //Compute array of [(x1-mean)^2, (x2-mean)^2, (x3-mean)^2, ... ]
    int *array_of_difference_between_mean_and_elements_device;
    cudaMalloc(&array_of_difference_between_mean_and_elements_device, size);
    compute_difference_between_mean_and_elements<<<grid_size, block_size>>>(array_of_difference_between_mean_and_elements_device, device_input, mean);

    //Compute (x1-mean)^2 + (x2 - mean) ^ 2 + ...
    int *output_array_for_sum_of_difference_between_elements, *output_array_for_sum_of_difference_between_elements_host;
    output_array_for_sum_of_difference_between_elements_host = (int *)malloc(blocks * sizeof(int));
    cudaMalloc(&output_array_for_sum_of_difference_between_elements, blocks * sizeof(int));
    reduce<<<grid_size, block_size, maxThreads * sizeof(int)>>>(array_of_difference_between_mean_and_elements_device, output_array_for_sum_of_difference_between_elements);
    cudaMemcpy(output_array_for_sum_of_difference_between_elements_host, output_array_for_sum_of_difference_between_elements, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 1;i < blocks;i ++){
        output_array_for_sum_of_difference_between_elements_host[0] += output_array_for_sum_of_difference_between_elements_host[i];
    }
    
    // Compute variance i.e ((x1 - mean)^2 + (x2 - mean)^2 ...) / n
    output_array_for_sum_of_difference_between_elements_host[0] = output_array_for_sum_of_difference_between_elements_host[0] / n;
    cout << endl << "Variance from GPU : " << output_array_for_sum_of_difference_between_elements_host[0] << endl;

    //Compute square root of (x1 - mean) ^ 2 + (x2 - mean) ^ 2 ...
    output_array_for_sum_of_difference_between_elements_host[0] = sqrt(output_array_for_sum_of_difference_between_elements_host[0]);

    cout << endl << "Standard deviation from  GPU : " << output_array_for_sum_of_difference_between_elements_host[0] << endl;

}