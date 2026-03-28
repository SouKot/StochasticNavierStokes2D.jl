using BenchmarkTools
using Printf

include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D
using .StochasticNavierStokes2D.Types: Grid, Parameters, ModelState, EnsembleStatistics
using .StochasticNavierStokes2D.Stochastics: init_initial_condition!
using .StochasticNavierStokes2D.Poisson: PoissonSolver
using .StochasticNavierStokes2D.Solver: solve_spde!

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    using CUDA
end

function run_benchmark(Nx, Ny, Ne, Nt; scheme=:Euler)
    println("--- Benchmarking Scheme: $scheme | Grid: $(Nx)x$(Ny) | Ensemble: $Ne ---")
    
    # Initialize Types based on backend precision
    if USE_GPU
        T_prec = Float32
        grid = Grid(Nx, Ny; array_type=CuArray, T=T_prec)
        params = Parameters(; dt = 0.005, T_end = Nt * 0.005, T=T_prec) 
        state = ModelState(Nx, Ny, Ne; array_type=CuArray, T=T_prec)
        stats = EnsembleStatistics(Nx, Ny; array_type=CuArray, T=T_prec)
    else
        T_prec = Float64
        grid = Grid(Nx, Ny; array_type=Array, T=T_prec)
        params = Parameters(; dt = 0.005, T_end = Nt * 0.005, T=T_prec) 
        state = ModelState(Nx, Ny, Ne; array_type=Array, T=T_prec)
        stats = EnsembleStatistics(Nx, Ny; array_type=Array, T=T_prec)
    end
    
    # Create solver
    psolver = PoissonSolver(state.omega, grid)
    
    # Initialize IC
    init_initial_condition!(state.omega, grid, params)
    
    # Pre-compile run (1 step)
    params_compile = Parameters(; dt = 0.005, T_end = 0.005)
    solve_spde!(state, grid, params_compile, stats, psolver; save_interval=1.0, scheme=scheme)
    
    # Benchmark Function Wrapper
    function step_n_times()
        params_run = Parameters(; dt = 0.005, T_end = Nt * 0.005)
        solve_spde!(state, grid, params_run, stats, psolver; save_interval=1000.0, scheme=scheme)
    end
    
    println("Profiling $Nt steps...")
    b = @benchmark $step_n_times() samples=5 evals=1
    
    display(b)
    
    med_time = median(b).time / 1e9 # seconds
    sps = Nt / med_time
    
    # Since it's batched, total individual grid steps per second:
    total_sps = (Nt * Ne) / med_time
    
    @printf("\n>> Performance: %.2f Batched Steps/sec\n", sps)
    @printf(">> Equivalent to: %.2f Individual Grid Steps/sec\n\n", total_sps)
    
    return b
end

# Main entry point for the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    # Warmup parameters
    Nx, Ny = 128, 128
    Ne = 10
    Nt = 10 # Number of steps to profile per sample
    
    println("Using $(Threads.nthreads()) CPU threads.")
    
    # Benchmark Euler
    run_benchmark(Nx, Ny, Ne, Nt, scheme=:Euler)
    
    # Benchmark Crank-Nicolson
    run_benchmark(Nx, Ny, Ne, Nt, scheme=:CrankNicolson)
end
