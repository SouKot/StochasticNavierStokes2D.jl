#!/bin/bash

# A simple script to benchmark the Solver at different CPU core counts
echo "Starting CPU Scaling Benchmarks..."

for cores in 1 2 4 8; do
    echo ""
    echo "=================================================="
    echo " Running benchmark with $cores CPU Threads"
    echo "=================================================="
    julia --project=. -t $cores test/benchmark.jl
done

# Note on GPU Benchmarking:
# To benchmark the GPU, you would need to change the @init_parallel_stencil
# commands in src/Stochastics.jl and src/Kernels.jl to use CUDA instead of Threads,
# and ensure you are passing CuArrays to the solver instead of standard Arrays.
