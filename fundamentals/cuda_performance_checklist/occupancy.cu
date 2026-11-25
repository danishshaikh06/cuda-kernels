
/*
===============================================================================
NCU PROFILING GUIDE: BLOCK SIZE OPTIMIZATION AND GPU OCCUPANCY
===============================================================================

OBJECTIVE:
Analyze how block size affects GPU resource utilization and occupancy
for the copyDataCoalesced() kernel.

Current configuration:
- Block Size = 1024
- Grid Size = calculated based on N
- Coalesced global memory access
- N = 1 << 24

Experiment:
Compare different block sizes such as:
128, 256, 512, 1024.

IMPORTANT NCU SECTIONS AND METRICS:

1. GPU Speed Of Light Throughput
   - Duration
   - Memory Throughput
   - DRAM Throughput
   - L1/TEX Cache Throughput
   - L2 Cache Throughput
   - SM Active Cycles

   PURPOSE:
   Determine how block size affects memory throughput and
   overall kernel execution time.

2. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Observe how block size affects the number of active and
   eligible warps available to each scheduler.

3. Occupancy
   - Block Limit SM
   - Block Limit Registers
   - Block Limit Shared Mem
   - Block Limit Warps
   - Theoretical Active Warps per SM
   - Theoretical Occupancy
   - Achieved Occupancy
   - Achieved Active Warps Per SM

   PURPOSE:
   Identify how block size affects theoretical and achieved
   GPU occupancy.

4. Launch Statistics
   - Block Size
   - Grid Size
   - Threads
   - Registers Per Thread
   - Shared Memory Configuration Size
   - Static Shared Memory Per Block
   - Dynamic Shared Memory Per Block
   - Waves Per SM
   - # SMs

   PURPOSE:
   Compare launch configurations and determine whether the
   grid provides enough blocks to utilize the available SMs.

5. Compute Workload Analysis
   - Executed Ipc Active
   - Executed Ipc Elapsed
   - Issued Ipc Active
   - Issue Slots Busy
   - SM Busy

   PURPOSE:
   Examine whether different block sizes improve SM utilization
   and instruction execution efficiency.

6. Memory Workload Analysis
   - Memory Throughput
   - Mem Busy
   - Max Bandwidth
   - L1/TEX Hit Rate
   - L2 Hit Rate
   - Mem Pipes Busy
   - Local Memory Spilling Requests

   PURPOSE:
   Determine whether block size affects memory subsystem
   utilization or introduces register spilling.

7. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp
   - Avg. Not Predicated Off Threads Per Warp

   PURPOSE:
   Observe warp execution efficiency and thread utilization.

EXPECTED OBSERVATIONS:

SMALL BLOCK SIZE (128):
- More blocks are required.
- More scheduling units are available across the grid.
- More blocks may be resident on each SM, depending on resources.
- Occupancy depends on register and warp limits.

MEDIUM BLOCK SIZE (256 OR 512):
- Different balance between block residency and scheduling.
- May provide good occupancy and resource utilization.
- Performance depends on GPU architecture and workload.

LARGE BLOCK SIZE (1024):
- Maximum CUDA threads per block on many NVIDIA GPUs.
- Fewer blocks are required for the same N.
- May limit the number of resident blocks per SM.
- May reduce scheduling flexibility.
- Can still achieve good occupancy if resource requirements allow.

PRIMARY METRICS:
1. Duration
2. Theoretical Occupancy
3. Achieved Occupancy
4. Achieved Active Warps Per SM
5. Eligible Warps Per Scheduler
6. Issued Warp Per Scheduler
7. Waves Per SM
8. Memory Throughput
9. SM Busy

IMPORTANT EXPERIMENT NOTES:

- Keep N constant for every experiment.
- Change only Block Size.
- Recalculate Grid Size whenever Block Size changes.
- Keep the kernel and memory access pattern identical.
- Compare NCU reports for each configuration.
- Check Block Limit Registers and Block Limit Warps.
- Check Waves Per SM to understand grid size and available parallelism.

IMPORTANT:
A larger block size does not automatically produce higher occupancy
or better performance.

Theoretical Occupancy and Achieved Occupancy are different.
Always compare both when evaluating block size optimization.

The cudaOccupancyMaxPotentialBlockSize() API can suggest a launch
configuration, but the suggested configuration is not guaranteed
to provide the best execution time.

===============================================================================
*/

#include <iostream>
#include <cuda_runtime.h>

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
    const int n = 1 << 24; // Adjust the data size for workload
    float *in, *out;

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blockSize = 1024; // Optimal block size for many devices
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate the number of blocks

    // Optimize grid dimensions based on device properties
    int minGridSize = 40;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copyDataCoalesced, 0, 0);

    // Print suggested block size and minimum grid size
    std::cout << "Recommended block size: " << blockSize
              << ", Minimum grid size: " << minGridSize << std::endl;

    numBlocks = (n + blockSize - 1) / blockSize;

    // Launch coalesced kernel
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

    return 0;
}