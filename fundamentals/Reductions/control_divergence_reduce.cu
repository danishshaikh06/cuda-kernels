#include <iostream>
#include <cuda.h>

__global__ void FixDivergenceKernel(float* input, float* output) {
    unsigned int i = threadIdx.x; //threads start next to each other
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) { // furthest element is blockDim away
        if (threadIdx.x < stride) { // 
            input[i] += input[i + stride]; // each thread adds a distant element to its assigned position
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
}

int main() {
    const int size = 2048;
    const int bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float;

    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; 
    }

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    FixDivergenceKernel<<<1, size / 2>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Sum is " << *h_output << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}