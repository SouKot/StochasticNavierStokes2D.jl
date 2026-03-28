# StochasticNavierStokes2D.jl

> [!NOTE]
> **AI-Assisted Development & Auditing**
> This framework was developed in tandem with advanced agentic AI assistance. However, this is not generic AI-generated code. Every single algorithmic step, Pseudo-Spectral memory allocation, out-of-core batching loop, has been rigorously audited, benchmarked, and mathematically verified by the primary author for exact computational fluid dynamics precision.

## The Physics Problem

This codebase solves the **2D Stochastic Navier-Stokes Equations** formulated within the Vorticity-Streamfunction ($\omega - \psi$) domain. It simulates turbulent fluid flows physically driven by independent continuous stochastic forcing vectors ($W_1, W_2$). 

The ultimate operational goal of the framework is to execute massive Monte Carlo ensemble simulations (e.g., $N_e = 12,800$ independent trajectories) to precisely compute the transient statistical dynamics:
- **Mean Field**: The deterministic expected evolution of the turbulent flow.
- **Variance**: The spread of physical uncertainty.
- **Third Moment (Skewness)**: Evaluation of asymmetrical velocity bursts.
- **Excess Kurtosis**

## Available Solvers & Numerics

The code features three unique explicit solvers capable of tackling varying degrees of mathematical stiffness:

1. **`Euler`**: A standard baseline Explicit Euler-Maruyama solver utilizing Finite Differences for strict spatial Jacobian advection evaluation.
2. **`CrankNicolson`**: Matches Explicit advection with a Semi-Implicit linear diffusion tracking envelope to increase explicit stability.
3. **`Spectral` (Recommended)**: It computes the explicit spatial derivatives natively in Fourier Space globally (Pseudo-Spectral Advection), and directly applies an exact energy-conserving Backward Euler implicit continuous diffusion mask natively before projecting back to physical space.

## Parallelization & HPC Architecture

This codebase utilizes two parallelization strategies:

### 1. Vectorized Ensemble Dispatch (CPU/GPU)
Every single explicit operation is mathematically broadcasted across a pure continuous 3-Dimensional array `(Nx, Ny, Ne)`.
- **CPU Threads (`Array`)**: Will automatically map calculations physically across 100% of available processor cores by reading standard Julia scalar Arrays.
- **GPU Dominance (`CuArray`)**: You can specifically trigger `USE_GPU=true`. The system allocates the identical structured arrays isolated entirely inside NVIDIA VRAM, allowing thousands of physical trajectories to compile simultaneously under CUDA.

### 2. Out-of-Core Memory Batching (The HDF5 Pipeline)
To simulate 12,800 high-resolution trajectories, the local memory footprint demands approximately 15.6 GiB (bigger than author computer's VRAM). To bypass standard desktop RAM/VRAM physical architectural limits, the codebase utilizes explicit **HDF5 Out-of-Core Batching**.
Instead of crashing your hardware due to Out-Of-Memory errors, the `scripts/run_simulation.jl` script natively:
1. Dispatches independent dynamic subsets of $N_{batch}$ physical samples (e.g., evaluating just $3,200$ loops per batch).
2. Explicitly integrates each subset independently over the entire timeframe.
3. Stores the resulting massive spatial arrays directly to disk via `.h5` files natively without persisting them in active VRAM.
4. Performs a final secondary sequential read-pass over the exported HDF5 batch files strictly to sum and compute unbiased global standard central moments (Mean, Variance, Skewness, Kurtosis), combining the statistics without holding the full 15.6 GiB tensor in live memory.
