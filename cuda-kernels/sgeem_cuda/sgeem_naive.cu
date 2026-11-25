#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(err) << "\n"; \
        return -1; \
    } \
} while (0)

__global__ void matrixMulSimple(float* A, float* B, float* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < width) {
        float value = 0;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

int main() {

    int width = 32;

    size_t size = width * width * sizeof(float);

    // Allocate host memory
    float *h_A = new float[width * width];
    float *h_B = new float[width * width];
    float *h_C = new float[width * width];

    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define thread block dimensions
    dim3 threads(16,16);

    // Define grid dimensions
    dim3 blocks(
        (width + threads.x - 1) / threads.x, //2
        (width + threads.y - 1) / threads.y //2 -> (total 4 blocks)
    );

    // Launch kernel
    matrixMulSimple<<<blocks, threads>>>(d_A, d_B, d_C, width);

    CUDA_CHECK(cudaGetLastError());       // Catches invalid launch configs immediately
    CUDA_CHECK(cudaDeviceSynchronize());  // Catches async/runtime kernel errors

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result matrix (first 4x4):\n";

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_C[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}