
/*
===============================================================================
NCU PROFILING GUIDE: COALESCED vs NON-COALESCED GLOBAL MEMORY ACCESS
===============================================================================

OBJECTIVE:
Compare global memory access efficiency between:
1. copyDataNonCoalesced()
2. copyDataCoalesced()

IMPORTANT NCU SECTIONS AND METRICS:

1. GPU Speed Of Light Throughput
   - Memory Throughput
   - DRAM Throughput
   - L1/TEX Cache Throughput
   - L2 Cache Throughput
   - Duration

   PURPOSE:
   Compare memory subsystem utilization and kernel execution time.

2. Memory Workload Analysis
   - Memory Throughput
   - Mem Busy
   - Max Bandwidth
   - L1/TEX Hit Rate
   - L2 Hit Rate
   - Mem Pipes Busy

   PURPOSE:
   Observe memory bandwidth utilization and cache behavior.

3. Memory Workload Analysis Tables
   - Examine global load/store memory access efficiency.
   - Check memory sectors utilized per thread.
   - Look for NCU OPT messages about uncoalesced global accesses.

   PURPOSE:
   Identify inefficient global memory transactions caused by
   non-coalesced access patterns.

4. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Observe whether memory access behavior affects warp scheduling
   and the availability of eligible warps.

5. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp

   PURPOSE:
   Examine warp execution efficiency.

6. Launch Statistics
   - Block Size
   - Grid Size
   - Threads
   - Waves Per SM

   PURPOSE:
   Ensure both kernels use comparable launch configurations.

EXPECTED OBSERVATIONS:

NON-COALESCED:
- More memory sectors may be required for global loads.
- Lower memory access efficiency.
- Potentially increased memory traffic.
- Potentially lower effective memory throughput.

COALESCED:
- Consecutive threads access consecutive memory locations.
- More efficient global memory transactions.
- Better utilization of memory sectors.
- Potentially improved memory throughput and lower Duration.

PRIMARY METRICS:
1. Duration
2. Memory Throughput
3. DRAM Throughput
4. L1/TEX Cache Throughput
5. L2 Hit Rate
6. Memory Workload Analysis Tables

IMPORTANT:
- Keep Block Size and Grid Size identical.
- Compare kernels individually.
- Cache behavior can influence the observed DRAM Throughput.
- Check NCU OPT messages for uncoalesced global accesses.

===============================================================================
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[(index * 2) % n];
    }
}

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

void initializeArray(float *arr, int n) {
    for(int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

int main() {
    const int n = 1 << 24; // Increase n to have a larger workload
    float *in, *out;

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blockSize = 128; // Define block size
    // int blockSize = 1024; // change this when talking about occupancy
    int numBlocks = (n + blockSize - 1) / blockSize; // Ensure there are enough blocks to cover all elements

    // Launch non-coalesced kernel
    copyDataNonCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    initializeArray(out, n); // Reset output array

    // Launch coalesced kernel
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

    return 0;
}
