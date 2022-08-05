#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include "cublas_v2.h"

using namespace std;

int main(void){
    //shijian
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    float *A, *B, *C;
    int m, n, k;
    float alpha=1, beta=0;
    m = 4096;
    n = 4096;
    k = 4096;
    float *D_A, *D_B, *D_C;

    A = (float*)malloc(m * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C = (float*)malloc(m * n * sizeof(float));

    cudaMalloc((void **)&D_A, m * k * sizeof(float));
    cudaMalloc((void **)&D_B, m * k * sizeof(float));
    cudaMalloc((void **)&D_C, m * k * sizeof(float));
    
    //cublas
    cublasHandle_t handle; 

    //赋值
    for(int i=0;i<m*k;++i){
        A[i] = 2.0;
    }
    for(int i=0;i<k*n;++i){
        B[i] = 2.0;
    }

    cublasCreate(&handle);

    cublasSetMatrix(m, k, sizeof(float), A, m, D_A, m);
    cublasSetMatrix(k, n, sizeof(float), B, k, D_B, k);    
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, D_A, m, D_B, k, &beta, D_C, m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time, flops;
    cudaEventElapsedTime(&time, start, stop);

    flops = m * n * (k * 2.0) / (time / 1000) / (1000000000);
	
	std::cout<<"Time is "<<time<<std::endl;
    std::cout<<"Flops is "<<flops<<std::endl;

    cublasGetMatrix(m, n, sizeof(float), D_C, m, C, m);

    cout<<C[0]<<endl;

    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(D_C);

    free(A);
    free(B);
    free(C);

    cublasDestroy(handle);

}