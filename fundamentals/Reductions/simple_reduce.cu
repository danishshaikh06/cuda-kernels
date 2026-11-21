#include<iostream>
#include<cuda.h>

__global__ void SimpleReductionKernel(float *input, float *output){
    unsigned int i = threadIdx.x * 2;
    for (unsigned int stride = 1; stride<=blockDim.x; stride*=2){
        if (threadIdx.x % stride == 0){
            input[i] += input[i+1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        *output = input[0];
    }
}

int main(){
   const int size = 2048;
   const int bytes = size * sizeof(float);
}
