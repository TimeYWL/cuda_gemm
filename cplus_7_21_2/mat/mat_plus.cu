#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

#define Row 64
#define Col 64

struct Matrix
{
    /* data */
    int width = Col;
    int height = Row;
    int *elements;
};

__global__ void mat_plus(Matrix *a, Matrix *b, Matrix *c){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int val_a, val_b, val_c;
    
    
    val_a = a->elements[row*(a->width)+col];
    val_b = b->elements[row*(b->height)+col];
    val_c = val_a + val_b;
    c->elements[row*(c->height)+col] = val_c;
    
}

__global__ void test_fuc(){
    printf("%d", threadIdx.x);
}

int main(){
    int nBytes = Row * Col * sizeof(int);
    int test = 0;
    Matrix *a, *b, *c;
    cudaMallocManaged((void **)&a, sizeof(Matrix));
    cudaMallocManaged((void **)&b, sizeof(Matrix));
    cudaMallocManaged((void **)&c, sizeof(Matrix));
    cudaMallocManaged((void **)&a->elements, nBytes);
    cudaMallocManaged((void **)&b->elements, nBytes);
    cudaMallocManaged((void **)&c->elements, nBytes);

    for(int i=0; i<Row*Col; i++){
        a->elements[i] = 2;
        b->elements[i] = 8;
    }
    a->height = Row;
    a->width = Col;
    b->height = Row;
    b->width = Col;
    c->height = Row;
    c->width = Col;

    dim3 blocksize;
    blocksize.x = 32;
    blocksize.y = 32;
    dim3 gridsize;
    gridsize.x = (a->height + blocksize.x-1)/blocksize.x;
    gridsize.y = (a->width + blocksize.y-1)/blocksize.y;
    

    mat_plus<<<gridsize, blocksize>>>(a, b, c);

    cudaDeviceSynchronize();


    float maxError = 0;
    for(int i=0;i<a->height*a->width;i++){
        maxError = max(fabs(c->elements[i]-10), maxError);
    }
    std::cout << "最大误差: " << maxError << std::endl;

    //shifang neicun 
    cudaFree(a);
    cudaFree(b);
    cudaFree(c); 

    
    return 0;
}
