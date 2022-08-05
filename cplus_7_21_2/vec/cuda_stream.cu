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

    //创建cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    int N = 1024 * 1024 * 100;
    int nBytes = N * sizeof(float);
    float *x, *y, *z;
    float *d_x, *d_y, *d_z;

    //gpu 内存分配
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    //cpu 内存锁定
    cudaHostAlloc((void **)&x, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&y, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&z, nBytes, cudaHostAllocDefault);

    for(int i =0; i<N; ++i){
        x[i] = 10.0;
        y[i] = 20.0;
    }

    //dingyi kernel peizhi
    dim3 blockSize(1024);
    dim3 gridSize(N / 1024);

    gettimeofday( &start, NULL );

    //使用stream 数据传输与计算同步进行
    for (int i = 0; i < N; i += N/25){

		cudaMemcpyAsync(d_x, x + i, nBytes, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_y, y + i, nBytes, cudaMemcpyHostToDevice, stream);
 
		add<<<gridSize, blockSize, 0, stream >>>(d_x, d_y, d_z, N);
 
		cudaMemcpyAsync(z + i, d_z, nBytes, cudaMemcpyDeviceToHost, stream);
    }

    // 同步device 保证结果能正确访问
    cudaStreamSynchronize(stream);

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