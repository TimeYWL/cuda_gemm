#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

using namespace std;

__global__ void add(float* x, float* y, float* z, int n){
    //huoqu suoyin
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    //buchang
    z[index] = x[index] + y[index];
    
}

int main(){

    //shijian
    struct timeval start, end;

    int N = 1024 * 1024 *100;
    int nBytes = N * sizeof(float);
    //shenqing host neicun
    float *x, *y, *z;    

    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    for(int i =0; i<N; ++i){
        x[i] = 10.0;
        y[i] = 20.0;
    }

    //dingyi kernel peizhi
    dim3 blockSize(1024);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x);

    gettimeofday( &start, NULL );

    //zhixing kernel
    add<<<gridSize, blockSize>>>(x, y, z, N);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);
    
    std::cout<<z[0]<<endl;

    //shifang neicun 
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}