#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define BLOCK_SIZE 16
#define N 100

__global__ void gpu_matrix_mul(int *a, int *b, int *c){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if(col < N && row < N){
        for(int i = 0;i < N; i++){
            sum += a[row*N + i] * b[i*N + col];
        }
        c[row*N + col] = sum;
    }
}

void cpu_matrix_mul(int A[N][N], int B[N][N], int C[N][N]){
    
    auto start = high_resolution_clock::now();
    for(int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            for(int k = 0;k < N;k ++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(stop - start).count();
    cout << endl << " CPU exec time: " << cpu_time << endl;
}

int main(){

    //CPU duration count
    int CPU_A[N][N], CPU_B[N][N], CPU_C[N][N];

    for(int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            CPU_A[i][j] = rand()%293;
            CPU_B[i][j] = rand()%66;
        }
    }

    cpu_matrix_mul(CPU_A, CPU_B, CPU_C);

    //GPU duration count
    int *host_a, *host_b, *host_c, *device_a, *device_b, *device_c;

    host_a = (int *)malloc((N*N) * sizeof(int));
    host_b = (int *)malloc((N*N) * sizeof(int));
    host_c = (int *)malloc((N*N) * sizeof(int));
    
    for(int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            host_a[i * N + j] = CPU_A[i][j];
            host_b[i * N + j] = CPU_B[i][j];
        }
    }

    cudaMalloc(&device_a, (N*N)*sizeof(int));
    cudaMalloc(&device_b, (N*N)*sizeof(int));
    cudaMalloc(&device_c, (N*N)*sizeof(int));

    cudaMemcpy(device_a, host_a, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, (N*N)*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    auto start = high_resolution_clock::now();
    gpu_matrix_mul<<<dimGrid, dimBlock>>>(device_a, device_b, device_c);
    auto stop = high_resolution_clock::now();
    auto gpu_time = duration_cast<microseconds>(stop - start).count();
    cout << endl << " GPU time: " << gpu_time << endl;

    cudaMemcpy(host_c, device_c, (N*N)*sizeof(int), cudaMemcpyDeviceToHost);

    //Verify 

    cout << host_c[0] << " " << CPU_C[0][0] << endl;
    cout << host_c[1] << " " << CPU_C[0][1] << endl;

    for(int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            if(host_c[i * N + j] != CPU_C[i][j]){
                cout << endl << "FAILED" << endl;
                return -1;
            }
        }
    }

    cout << endl << "PASSED" << endl;

}