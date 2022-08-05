//Example. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "cublas_v2.h"
#define M 4096 //6行
#define N 4096 //5列
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/*一个把坐标（i, j）转化为cublas数据格式坐标的函数；
假设一个点位于第i行第j列，每一列的长度为ld，
则计算出来的列优先坐标就是 j*ld+i */

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q,
 float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int main (void){

    //shijian
    struct timeval start, end;    
    gettimeofday( &start, NULL );

    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle; //管理一个cublas上下文的句柄
    
    int i, j;
    float* devPtrA;
    float* a = 0;
    
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }   

    
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    cudaFree (devPtrA);
    
    cublasDestroy(handle);
    
    free(a);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);
    
    return EXIT_SUCCESS;
}