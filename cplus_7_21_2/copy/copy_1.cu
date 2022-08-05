#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

#define BLOCK_SIZE 32

__global__ void test_fun(){
    printf("%d;", blockDim.x);
    printf("%d;", blockDim.y);
}

// using ILP 2 to improve the performance
__global__ void matrixMulSharedILPkernel(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int n, int k){
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val[2] = {0.0f};

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int iter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(int i = 0; i < iter; i++)
    {
        // read data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[row * k + i * BLOCK_SIZE + threadIdx.x];
        shTileA[threadIdx.y + BLOCK_SIZE/2][threadIdx.x] = fpMatrixA[(row + BLOCK_SIZE/2) * k + i * BLOCK_SIZE + threadIdx.x];

        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + col];
        shTileB[threadIdx.y + BLOCK_SIZE/2][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y + BLOCK_SIZE/2) * n + col];

        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++){
            val[0] += shTileA[threadIdx.y][j] * shTileB[j][threadIdx.x];
            val[1] += shTileA[threadIdx.y + BLOCK_SIZE/2][j] * shTileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    fpMatrixC[row * n + col] = val[0];
    fpMatrixC[(row + BLOCK_SIZE/2) * n + col] = val[1];
}

__global__ void matrixMulSharedKernel_op1(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int n, int k){
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;
    int i, l;

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int nIter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(i = 0; i < nIter; i++){
        // load data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + nCol];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for(l = 0; l < BLOCK_SIZE; l++){
            fCVal += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }
    // store results into global memory
    fpMatrixC[nRow * n + nCol] = fCVal;
}

__global__ void matrixMulGlobalKernel(float* pfMatrixA, float* pfMatrixB, float* pfMatrixC, int m, int n, int k){
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    for(int i =0; i < k; i++)
    {
        fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
    }

    pfMatrixC[nRow * n + nCol] = fCVal;
}

int main(){

    //shijian
    struct timeval start, end;

    float *fpMatrixA, *fpMatrixB, *fpMatrixC, *test;
    int m, n, k;
    m = 4096;
    n = 4096;
    k = 4096;

    cudaMallocManaged((void **) &fpMatrixA, m*k*sizeof(float));
    cudaMallocManaged((void **) &fpMatrixB, k*n*sizeof(float));
    cudaMallocManaged((void **) &fpMatrixC, m*n*sizeof(float));
    cudaMallocManaged((void **) &test, sizeof(float));

    
    for(int i=0;i<m*k;++i){
        fpMatrixA[i] = 1;
    }
    for(int i=0;i<k*n;++i){
        fpMatrixB[i] = 1;
    }

    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = BLOCK_SIZE;
    blockSize.y = BLOCK_SIZE;
    gridSize.x = (m + blockSize.x-1)/blockSize.x;
    gridSize.y = (n + blockSize.y-1)/blockSize.y;
    //blockSize.y = BLOCK_SIZE/2;

    gettimeofday(&start, NULL);

    //matrixMulSharedKernel_op1<<<gridSize, blockSize>>>(fpMatrixA, fpMatrixB, fpMatrixC, m, n, k);
    matrixMulGlobalKernel<<<gridSize, blockSize>>>(fpMatrixA, fpMatrixB, fpMatrixC, m, n, k);
    //matrixMulSharedILPkernel<<<gridSize, blockSize>>>(fpMatrixA, fpMatrixB, fpMatrixC, m, n, k);
    //test_fun<<<gridSize, blockSize>>>();

    cudaDeviceSynchronize();

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    std::cout<<fpMatrixC[0]<<std::endl;
    
    //shifang neicun 
    cudaFree(fpMatrixC);
    cudaFree(fpMatrixB);
    cudaFree(fpMatrixA);

    return 0;
}