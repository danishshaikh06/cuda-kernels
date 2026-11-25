
/*
===============================================================================
NCU PROFILING GUIDE: DATA PRIVATIZATION (GLOBAL MEMORY vs SHARED MEMORY)
===============================================================================

OBJECTIVE:
Compare:
1. windowSumDirect()
2. windowSumPrivatized()

DIRECT KERNEL:
- Reads window elements directly from global memory.
- Multiple threads may access overlapping input regions.
- Relies on the GPU cache hierarchy to handle data reuse.

PRIVATIZED KERNEL:
- Loads input elements into shared memory.
- Threads reuse data stored in shared memory.
- Uses shared memory to reduce repeated global memory accesses.

IMPORTANT NCU SECTIONS AND METRICS:

1. GPU Speed Of Light Throughput
   - Duration
   - Memory Throughput
   - DRAM Throughput
   - L1/TEX Cache Throughput
   - L2 Cache Throughput
   - Compute (SM) Throughput
   - SM Active Cycles

   PURPOSE:
   Compare kernel execution time and memory subsystem utilization.

2. Memory Workload Analysis
   - Memory Throughput
   - Mem Busy
   - Max Bandwidth
   - L1/TEX Hit Rate
   - L2 Hit Rate
   - Mem Pipes Busy
   - Local Memory Spilling Requests
   - Shared Memory Spilling Requests

   PURPOSE:
   Determine whether privatization reduces global memory traffic
   and improves memory access efficiency.

   Compare L1/TEX Cache Throughput and L1/TEX Hit Rate to understand
   how the direct kernel benefits from cache reuse.

3. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Observe whether shared memory usage affects warp scheduling
   and the availability of eligible warps.

4. Occupancy
   - Block Limit SM
   - Block Limit Registers
   - Block Limit Shared Mem
   - Block Limit Warps
   - Theoretical Active Warps per SM
   - Theoretical Occupancy
   - Achieved Occupancy
   - Achieved Active Warps Per SM

   PURPOSE:
   Determine whether shared memory allocation limits the number
   of active blocks and warps per SM.

   IMPORTANT:
   Pay particular attention to Block Limit Shared Mem.

5. Launch Statistics
   - Block Size
   - Grid Size
   - Threads
   - Registers Per Thread
   - Shared Memory Configuration Size
   - Static Shared Memory Per Block
   - Dynamic Shared Memory Per Block
   - Waves Per SM

   PURPOSE:
   Compare resource requirements and launch configurations.

6. Compute Workload Analysis
   - Executed Ipc Active
   - Executed Ipc Elapsed
   - Issued Ipc Active
   - Issue Slots Busy
   - SM Busy

   PURPOSE:
   Determine whether shared memory operations and synchronization
   affect instruction execution and SM utilization.

7. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp
   - Avg. Not Predicated Off Threads Per Warp

   PURPOSE:
   Examine warp execution efficiency and the effects of
   synchronization and boundary conditions.

8. Instruction Statistics
   - Executed Instructions
   - Issued Instructions
   - Avg. Executed Instructions Per Scheduler
   - Avg. Issued Instructions Per Scheduler

   PURPOSE:
   Compare instruction overhead between direct global memory
   accesses and shared memory privatization.

EXPECTED OBSERVATIONS:

DIRECT KERNEL:
- Repeated overlapping global memory accesses.
- Potentially more global load instructions.
- May benefit significantly from L1/TEX and L2 cache reuse.
- Does not require explicit shared memory allocation.
- May have lower synchronization overhead.

PRIVATIZED KERNEL:
- Loads input elements into shared memory.
- Reuses shared memory values across multiple threads.
- May reduce repeated global memory accesses.
- Requires shared memory allocation.
- Introduces __syncthreads() synchronization overhead.
- May reduce occupancy depending on shared memory requirements.

PRIMARY METRICS:
1. Duration
2. DRAM Throughput
3. Memory Throughput
4. L1/TEX Cache Throughput
5. L1/TEX Hit Rate
6. Shared Memory Configuration Size
7. Static Shared Memory Per Block
8. Block Limit Shared Mem
9. Theoretical Occupancy
10. Achieved Occupancy

IMPORTANT EXPERIMENT NOTES:

- Keep N identical for both kernels.
- Keep windowSize identical.
- Keep Block Size identical at 256 threads.
- Compare both kernels individually in NCU.
- Check shared memory consumption and occupancy.
- Examine L1/TEX and L2 cache behavior.

IMPORTANT CODE OBSERVATION:

The current privatized implementation only loads the elements belonging
to the current block into shared memory.

Threads at block boundaries cannot access neighboring block elements
through sharedData. Consequently, the privatized kernel does not
calculate the same complete sliding window as the direct kernel.

Additionally, the shared memory array is declared with 1024 floats,
but only blockDim.x elements are loaded.

The __syncthreads() call is inside the idx < n condition. This can
cause incorrect synchronization when the final block has inactive
threads.

Fix these correctness issues before treating the results as a valid
performance comparison.

A proper tiled implementation should load the required halo elements
from neighboring positions into shared memory and ensure all threads
participate in synchronization.

===============================================================================
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel without privatization: Direct global memory access
__global__ void windowSumDirect(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = idx + i;
            if (accessIdx >= 0 && accessIdx < n) {
                sum += input[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

// Kernel with privatization: Preload window elements into registers
__global__ void windowSumPrivatized(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    __shared__ float sharedData[1024]; // Assuming blockDim.x <= 1024

    // Load input into shared memory (for demonstration, assuming window fits into shared memory)
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
        __syncthreads(); // Ensure all loads are complete

        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = threadIdx.x + i;
            // Check bounds within shared memory
            if (accessIdx >= 0 && accessIdx < blockDim.x && (idx + i) < n && (idx + i) >= 0) {
                sum += sharedData[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1.0f; // Simple initialization for demonstration
    }
}

int main() {
    int n = 1<<20; // Example array size
    int windowSize = 5; // Example window size
    float *input, *output;
    float *d_input, *d_output;

    input = (float*)malloc(n * sizeof(float));
    output = (float*)malloc(n * sizeof(float));

    // Initialize input array
    initializeArray(input, n);

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernels
    windowSumDirect<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    windowSumPrivatized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}