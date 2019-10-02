#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define BLOCKSIZE 16
#define N 1000

__global__ void gpu_mat_vec_multiply(double *device_mat,
                                    double *device_vec,
                                    double *device_res){
    
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tindex = tidx + gridDim.x * BLOCKSIZE * tidy;

    if(tindex < N){
        int i; int m = tindex * N;
        device_res[tindex] = 0.0;
        for(int i = 0;i < N;i ++){
            device_res[tindex] += device_mat[m + i] * device_vec[i];
        }
    }

    __syncthreads();
}

int main(){

    double *host_mat, *host_vec, *host_res;
    double *device_mat, *device_vec, *device_res;

    host_mat = new double[N * N];
    host_vec = new double[N];
    host_res = new double[N];

    for(int i = 0;i < N;i ++){
        host_vec[i] = double(rand()%100);
        for(int j = 0;j < N;j ++){
            host_mat[i * N + j] = double(rand()%40);
        }
    }

    cudaMalloc(&device_mat, (N*N)*sizeof(double));
    cudaMalloc(&device_vec, N*sizeof(double));
    cudaMalloc(&device_res, N*sizeof(double));

    cudaMemcpy(device_mat, host_mat, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vec, host_vec, N*sizeof(double), cudaMemcpyHostToDevice);

    int max = BLOCKSIZE * BLOCKSIZE;
    int BLocksPerGrid = N / max + 1;
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    if(N % max == 0) BLocksPerGrid --;
    dim3 dimGrid(1, BLocksPerGrid);
    gpu_mat_vec_multiply<<<dimGrid, dimBlock>>>(device_mat, device_vec, device_res);
    

}