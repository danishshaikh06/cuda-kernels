
/*
===============================================================================
NCU PROFILING GUIDE: DATA PRIVATIZATION (WITHOUT vs WITH PRIVATIZATION)
===============================================================================

OBJECTIVE:
Compare:
1. vectorAdd()
2. vectorAddPrivatized()

UNDERSTANDING PRIVATIZATION:

Privatization means giving each thread its own private copy of the
data it needs for computation.

WITHOUT PRIVATIZATION:
- Threads directly access input arrays.
- Values are read from global memory.
- Computation is performed directly using loaded values.

WITH PRIVATIZATION:
- Each thread loads input values into private variables.
- Private variables are typically stored in registers.
- Computation is performed using these private values.

IMPORTANT:
In this example, each thread already reads unique input indices.
Therefore, the two kernels may generate identical machine instructions.
Privatization does not automatically improve performance.

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
   Compare execution time and memory subsystem utilization.

   EXPECTED:
   Both kernels may show nearly identical throughput because
   they perform the same global memory loads and addition.

2. Memory Workload Analysis
   - Memory Throughput
   - Mem Busy
   - Max Bandwidth
   - L1/TEX Hit Rate
   - L2 Hit Rate
   - Mem Pipes Busy
   - Local Memory Spilling Requests
   - Local Memory Spilling Request Overhead

   PURPOSE:
   Determine whether privatization changes global memory
   traffic or introduces local memory spilling.

   EXPECTED:
   No significant difference is expected if the compiler
   optimizes both implementations identically.

3. Launch Statistics
   - Block Size
   - Grid Size
   - Threads
   - Registers Per Thread
   - Waves Per SM

   PURPOSE:
   Compare launch configurations and register usage.

   IMPORTANT:
   Pay particular attention to Registers Per Thread.

   If privatization increases register usage, it may affect
   the number of active warps per SM.

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
   Determine whether changes in register usage affect
   theoretical and achieved occupancy.

5. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Examine whether register usage and resource availability
   affect warp scheduling.

6. Compute Workload Analysis
   - Executed Ipc Active
   - Executed Ipc Elapsed
   - Issued Ipc Active
   - Issue Slots Busy
   - SM Busy

   PURPOSE:
   Compare instruction execution efficiency and SM utilization.

7. Instruction Statistics
   - Executed Instructions
   - Issued Instructions
   - Avg. Executed Instructions Per Scheduler
   - Avg. Issued Instructions Per Scheduler

   PURPOSE:
   Check whether the compiler generates different instruction
   sequences for the two implementations.

8. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp
   - Avg. Not Predicated Off Threads Per Warp

   PURPOSE:
   Compare warp execution efficiency.

PRIMARY METRICS:
1. Duration
2. Registers Per Thread
3. Executed Instructions
4. Issued Instructions
5. Theoretical Occupancy
6. Achieved Occupancy
7. Eligible Warps Per Scheduler
8. Local Memory Spilling Requests

EXPECTED OBSERVATIONS:

WITHOUT PRIVATIZATION:
- Direct access to input arrays.
- Values are used in the addition operation.
- Global memory loads are required.

WITH PRIVATIZATION:
- Values are loaded into thread-private variables.
- Private variables may be stored in registers.
- May provide benefits when private values are reused.
- May increase register pressure in more complex workloads.

IMPORTANT EXPERIMENT NOTES:

- Keep N identical.
- Keep Block Size identical at 256 threads.
- Use identical input data.
- Compare both kernels individually in NCU.
- Check Registers Per Thread.
- Check Executed Instructions.
- Check Local Memory Spilling Requests.
- Check Duration.

IMPORTANT:
Privatization is useful when it reduces repeated memory accesses,
avoids contention, or enables data reuse.

In this particular vector addition example, each input value is
read only once per thread. The compiler may optimize both kernels
into identical machine instructions.

Use NCU Source Counters or generated SASS to verify whether
the two kernels actually differ.

===============================================================================
*/
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition without privatization
__global__ void vectorAdd(const float *a, const float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        result[index] = a[index] + b[index];
    }
}

// CUDA kernel for vector addition with privatization
__global__ void vectorAddPrivatized(const float *a, const float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float a_private = a[index]; // Load into private memory
        float b_private = b[index]; // Load into private memory
        result[index] = a_private + b_private;
    }
}

// Function to initialize the vectors with dummy data
void initData(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }
}

int main() {
    int n = 1<<20; // Size of the vectors
    float *a, *b, *result, *d_a, *d_b, *d_result;

    // Allocate memory on the host
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    result = (float*)malloc(n * sizeof(float));

    // Initialize vectors
    initData(a, n);
    initData(b, n);

    // Allocate memory on the device
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector addition kernel without privatization
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

    // Copy result back to host
    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the vector addition kernel with privatization
    vectorAddPrivatized<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

    // Copy result back to host
    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(a);
    free(b);
    free(result);

    return 0;
}