
/*
===============================================================================
NCU PROFILING GUIDE: WARP DIVERGENCE vs BRANCHLESS EXECUTION
===============================================================================

OBJECTIVE:
Compare:
1. processArrayWithDivergence()
2. processArrayWithoutDivergence()

The first kernel uses if-else conditional branching.
The second kernel uses arithmetic expressions instead of explicit
if-else branching.

IMPORTANT NCU SECTIONS AND METRICS:

1. GPU Speed Of Light Throughput
   - Duration
   - Compute (SM) Throughput
   - Memory Throughput
   - SM Active Cycles
   - L1/TEX Cache Throughput
   - L2 Cache Throughput

   PURPOSE:
   Compare execution time and GPU utilization between both kernels.

2. Compute Workload Analysis
   - Executed Ipc Active
   - Executed Ipc Elapsed
   - Issued Ipc Active
   - Issue Slots Busy
   - SM Busy

   PURPOSE:
   Determine whether branchless execution improves instruction
   execution efficiency and SM utilization.

3. Scheduler Statistics
   - One or More Eligible
   - Issued Warp Per Scheduler
   - No Eligible
   - Active Warps Per Scheduler
   - Eligible Warps Per Scheduler

   PURPOSE:
   Observe whether divergent execution affects warp scheduling
   and the availability of eligible warps.

4. Warp State Statistics
   - Warp Cycles Per Issued Instruction
   - Warp Cycles Per Executed Instruction
   - Avg. Active Threads Per Warp
   - Avg. Not Predicated Off Threads Per Warp

   PURPOSE:
   Examine how many threads actively execute instructions within
   each warp.

   IMPORTANT:
   Check Avg. Not Predicated Off Threads Per Warp to identify
   the effects of predication and conditional execution.

5. Instruction Statistics
   - Executed Instructions
   - Issued Instructions
   - Avg. Executed Instructions Per Scheduler
   - Avg. Issued Instructions Per Scheduler

   PURPOSE:
   Compare instruction execution overhead between conditional
   branching and branchless arithmetic.

6. Source Counters
   - Branch Instructions
   - Branch Instructions Ratio
   - Branch Efficiency
   - Avg. Divergent Branches

   PURPOSE:
   Identify branch divergence and understand how effectively
   threads execute conditional branches.

7. Launch Statistics
   - Block Size
   - Grid Size
   - Threads
   - Registers Per Thread
   - Waves Per SM

   PURPOSE:
   Ensure both kernels use comparable launch configurations.

8. Occupancy
   - Theoretical Occupancy
   - Achieved Occupancy
   - Achieved Active Warps Per SM
   - Block Limit Registers
   - Block Limit Warps

   PURPOSE:
   Check whether differences in register usage affect occupancy.

EXPECTED OBSERVATIONS:

WITH DIVERGENCE:
- Uses if-else conditional branching.
- Threads within a warp may follow different execution paths.
- Branch Efficiency may decrease when branches diverge.
- Avg. Divergent Branches may increase.
- Warp execution can require additional serialized work.

WITHOUT DIVERGENCE:
- Uses arithmetic expressions instead of explicit if-else.
- May reduce branch instructions.
- May reduce control-flow overhead.
- May introduce additional arithmetic instructions.
- May increase register usage or instruction count.

PRIMARY METRICS:
1. Duration
2. Branch Efficiency
3. Avg. Divergent Branches
4. Branch Instructions Ratio
5. Executed Instructions
6. Issued Instructions
7. Avg. Not Predicated Off Threads Per Warp
8. Issued Warp Per Scheduler

IMPORTANT EXPERIMENT NOTES:

- Keep N identical for both kernels.
- Keep Block Size identical at 256 threads.
- Compare both kernels individually in NCU.
- Use identical input data.
- Check Registers Per Thread for both kernels.
- Examine Source Counters to verify actual branch behavior.

IMPORTANT:
Branchless code does not automatically mean faster execution.
The compiler may convert conditional branches into predicated
instructions, and arithmetic expressions may increase instruction
count.

Check the generated SASS and Source Counters before concluding
that branch divergence has been eliminated.

===============================================================================
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void processArrayWithDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx] % 2 == 0) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] + 1;
        }
    }
}

__global__ void processArrayWithoutDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int isEven = !(data[idx] % 2);
        data[idx] = isEven * (data[idx] * 2) + (!isEven) * (data[idx] + 1);
    }
}

void benchmarkKernel(void (*kernel)(int *, int), int *data, int N, const char *kernelName) {
    int *devData;
    cudaMalloc(&devData, N * sizeof(int));
    cudaMemcpy(devData, data, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(devData, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s took %f milliseconds\n", kernelName, milliseconds);

    cudaMemcpy(data, devData, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 1 << 20; // Example size
    int *data = (int *)malloc(N * sizeof(int));

    // Initialize data
    for(int i = 0; i < N; i++) {
        data[i] = i;
    }

    benchmarkKernel(processArrayWithDivergence, data, N, "processArrayWithDivergence");
    benchmarkKernel(processArrayWithoutDivergence, data, N, "processArrayWithoutDivergence");

    free(data);
    return 0;
}