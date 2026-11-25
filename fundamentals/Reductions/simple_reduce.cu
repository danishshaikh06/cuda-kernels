#include<iostream>
#include<cuda.h>

__global__ void SimpleReductionKernel(float *input, float *output){
    unsigned int i = threadIdx.x * 2;
    for (unsigned int stride = 1; stride<=blockDim.x; stride*=2){
        if (threadIdx.x % stride == 0){
            input[i] += input[i+stride];
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

   float *h_input = new float[size]; // Dynamically allocates memory for an array and stores the address in h_input
   float *h_output = new float; // Dynamically allocates memory for one float and stores the address in h_output

   for(int i = 0; i<size; i++){
    h_input[i] = 1.0f;
   }

   float *d_input;
   float *d_output;

   cudaMalloc(&d_input, bytes);
   cudaMalloc(&d_output, sizeof(float));

   cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

   SimpleReductionKernel<<<1, size / 2>>>(d_input, d_output);

   cudaMemcpy(h_output, d_output, sizeof(float),cudaMemcpyDeviceToHost);

   std::cout<<"Sum is " << *h_output << std::endl;

   //clean
   delete[] h_input;
   delete h_output;
   cudaFree(d_input);
   cudaFree(d_output);

   return 0;
}


