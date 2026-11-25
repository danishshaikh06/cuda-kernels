
/*
===============================================================================
NCU PROFILING GUIDE: THREAD COARSENING (COARSENING FACTOR = 2)
===============================================================================

OBJECTIVE:
Compare normal vector addition (VecAdd) against thread-coarsened
vector addition (VecAddCoarsened).

NORMAL KERNEL:
- Each thread processes one element.

COARSENED KERNEL:
- Each thread processes two elements.
- Approximately half the number of threads are required.

IMPORTANT NCU SECTIONS AND METRICS:

1. GPU Speed Of Light Throughput
   - Duration
   - Compute (SM) Throughput
   - Memory Throughput
   - SM Active Cycles
   - SM Frequency

   PURPOSE:
   Compare execution time and GPU utilization between both kernels.

2. Compute Workload Analysis
   - Executed Ipc Active
   - Executed Ipc Elapsed
   - Issued Ipc Active
   - Issue Slots Busy
   - SM Busy

   PURPOSE:
   Determine whether thread coarsening improves instruction
   execution and SM utilization.

3. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Check whether increased work per thread affects warp scheduling
   and the availability of eligible warps.

4. Occupancy
   - Theoretical Active Warps per SM
   - Theoretical Occupancy
   - Achieved Occupancy
   - Achieved Active Warps Per SM
   - Block Limit Registers
   - Block Limit Warps

   PURPOSE:
   Determine whether thread coarsening changes GPU occupancy.

5. Launch Statistics
   - Block Size
   - Grid Size
   - Registers Per Thread
   - Threads
   - Waves Per SM

   PURPOSE:
   Compare launch configurations and register usage.

6. Instruction Statistics
   - Executed Instructions
   - Issued Instructions
   - Avg. Executed Instructions Per Scheduler
   - Avg. Issued Instructions Per Scheduler

   PURPOSE:
   Compare instruction execution overhead between the two kernels.

7. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp
   - Avg. Not Predicated Off Threads Per Warp

   PURPOSE:
   Observe warp execution efficiency and predication effects.

8. Memory Workload Analysis
   - Memory Throughput
   - Mem Busy
   - Max Bandwidth
   - L1/TEX Hit Rate
   - L2 Hit Rate
   - Local Memory Spilling Requests

   PURPOSE:
   Check whether increased work per thread affects memory behavior
   or causes register spilling.

EXPECTED OBSERVATIONS:

NORMAL VecAdd:
- One element processed per thread.
- More threads and blocks are required.
- Lower work per thread.

COARSENED VecAddCoarsened:
- Two elements processed per thread.
- Fewer threads and blocks are required.
- More work per thread.
- Potential reduction in indexing and scheduling overhead.
- Potential increase in Registers Per Thread.
- Achieved Occupancy may increase or decrease.

PRIMARY METRICS:
1. Duration
2. Achieved Occupancy
3. Theoretical Occupancy
4. Registers Per Thread
5. Eligible Warps Per Scheduler
6. Issued Warp Per Scheduler
7. Executed Instructions
8. SM Busy

IMPORTANT:
- Keep N and input data identical.
- Keep THREADS_PER_BLOCK identical.
- Coarsening factor = 2.
- Compare both kernels individually.
- Increase N because N = 1024 may be too small to expose
  meaningful performance differences.
- Check Waves Per SM to identify insufficient grid size.

===============================================================================
*/

#include <stdio.h>

#define N 1024
#define THREADS_PER_BLOCK 256 // This is just an example block size

// Original vector addition kernel without coarsening
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Vector addition kernel with thread coarsening
// Assuming a coarsening factor of 2
__global__ void VecAddCoarsened(float* A, float* B, float* C)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Coarsening factor applied here
    if (i < N)
        C[i] = A[i] + B[i];
    if (i + 1 < N) // Handle the additional element due to coarsening
        C[i + 1] = A[i + 1] + B[i + 1];
}

void random_init(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(float);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (float *)malloc(size); random_init(a, N);
    b = (float *)malloc(size); random_init(b, N);
    c = (float *)malloc(size);

    cudaEvent_t start, stop, startCoarsened, stopCoarsened;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCoarsened);
    cudaEventCreate(&stopCoarsened);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // warmup
    VecAdd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1) / (2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Start timer for VecAdd kernel
    cudaEventRecord(start);
    VecAdd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    // Wait for VecAdd kernel to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("VecAdd execution time: %f ms\n", milliseconds);

    // Start timer for VecAddCoarsened kernel
    cudaEventRecord(startCoarsened);
    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1) / (2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stopCoarsened);

    // Wait for VecAddCoarsened kernel to finish
    cudaEventSynchronize(stopCoarsened);

    float millisecondsCoarsened = 0;
    cudaEventElapsedTime(&millisecondsCoarsened, startCoarsened, stopCoarsened);
    printf("VecAddCoarsened execution time: %f ms\n", millisecondsCoarsened);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startCoarsened);
    cudaEventDestroy(stopCoarsened);

    return 0;
}