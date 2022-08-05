#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

#define BSIZE 64 //block_size
#define Q 16 //reg 方法中使用的寄存器数量，即单线程所需计算的C矩阵单元数量

__global__ void mat_times(float *A, float *B, float *C, int m, int n, int k){

    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;
    //从显存中取数据进行计算
    for(int i=0;i<k;i++){
        val += A[nRow * k + i] * B[n * i + nCol];
    }
    C[nRow * n + nCol] = val;
}

__global__ void mat_times_share(float *A, float *B, float *C, int m, int n, int k){
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float Shared_A[BSIZE][BSIZE];
    __shared__ float Shared_B[BSIZE][BSIZE];

    int nIter = (k + BSIZE - 1) / BSIZE;
    float tval = 0;

    for(int i=0;i<nIter;i++){
        Shared_A[threadIdx.y][threadIdx.x] = A[nRow * k + i * BSIZE + threadIdx.x];
        Shared_B[threadIdx.y][threadIdx.x] = B[(i * BSIZE + threadIdx.y) * n + nCol];

        __syncthreads();

        for(int j=0;j<BSIZE;j++){
            tval += Shared_A[threadIdx.y][j] * Shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[nRow * n + nCol] = tval;
}

__global__ void mat_times_share_reg(float *A, float *B, float *C, int m, int n, int k){
    
    const int q = Q;
    int nRow = blockIdx.y * blockDim.y * q + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float Share_A[BSIZE][BSIZE];
    __shared__ float Share_B[BSIZE][BSIZE];

    int nIter = (k + BSIZE - 1) / BSIZE;
    float tval[q] = {0};

    for(int i=0;i<nIter;i++){

        for(int p=0; p<q; p++){
            Share_A[threadIdx.y+BSIZE/q*p][threadIdx.x]=A[(nRow+BSIZE/q*p)*k+i*BSIZE+threadIdx.x];
            Share_B[threadIdx.y+BSIZE/q*p][threadIdx.x]=B[(i*BSIZE+threadIdx.y+BSIZE/q*p)*n+nCol];
        }

        __syncthreads();

        for(int j=0;j<BSIZE;j++){

            for(int p=0; p<q; p++){
                tval[p] += Share_A[threadIdx.y+BSIZE/q*p][j] * Share_B[j][threadIdx.x];
            }
        }

        __syncthreads();

    }

    for(int p=0; p<q; p++){
        C[(nRow+BSIZE/1*p)*n+nCol]=tval[p];
    }
}

int main(){

    //shijian
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    float *A, *B, *C;
    int m, n, k;
    m = 4096;
    n = 4096;
    k = 4096;
    float *D_A, *D_B, *D_C;
    
    //分别进行cpu和gpu的内存申请
    cudaMalloc((void **)&D_A, m * k * sizeof(float));
    cudaMalloc((void **)&D_B, m * k * sizeof(float));
    cudaMalloc((void **)&D_C, m * k * sizeof(float));

    A = (float*)malloc(m * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C = (float*)malloc(m * n * sizeof(float));

    for(int i=0;i<m*k;++i){
        A[i] = 2.0;
    }
    for(int i=0;i<k*n;++i){
        B[i] = 2.0;
    }
    //数据传输
    cudaMemcpy(D_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    //定义线程网大小
    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = BSIZE;
    blockSize.y = BSIZE;
    gridSize.x = (m + blockSize.x - 1) / blockSize.x;
    gridSize.y = (n + blockSize.y - 1) / blockSize.y;

	//记录时间
	cudaEventRecord(start);

    //global内存方法
    // mat_times<<<gridSize, blockSize>>>(D_A, D_B, D_C, m, n, k);
    //共享内存方法
    // mat_times_share<<<gridSize, blockSize>>>(D_A, D_B, D_C, m, n, k);
    //寄存器方法
    blockSize.y = BSIZE / Q;
    mat_times_share_reg<<<gridSize, blockSize>>>(D_A, D_B, D_C, m, n, k);

    //记录时间
    cudaEventRecord(stop);
    //等待device计算完成
	cudaEventSynchronize(stop);
    //数据传输和检查
    cudaMemcpy(C, D_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<10; i++)
        std::cout<<C[i]<<std::endl;

	//计算用时和浮点性能
	float time, flops;
	cudaEventElapsedTime(&time, start, stop);
    flops = m * n * (k * 2.0) / (time / 1000) / (1000000000);	
	std::cout<<"Time is "<<time<<std::endl;
    std::cout<<"Flops is "<<flops<<std::endl;

 

    //shifang neicun 
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
