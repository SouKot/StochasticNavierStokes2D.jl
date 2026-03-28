include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D
using .StochasticNavierStokes2D.Types: Grid, Parameters, ModelState, EnsembleStatistics
using .StochasticNavierStokes2D.Stochastics: init_initial_condition!
using .StochasticNavierStokes2D.Poisson: PoissonSolver
using .StochasticNavierStokes2D.Kernels
using .StochasticNavierStokes2D.Visualization

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    using CUDA
end

println("Running tests...")

# 1. Initialize Backend Arrays
Nx, Ny, Ne = 128, 128, 10

if USE_GPU
    T_prec = Float32
    grid = Grid(Nx, Ny; array_type=CuArray, T=T_prec)
    params = Parameters(; dt = 0.005, T_end = 0.05, T=T_prec) # very short run (10 steps)
    state = ModelState(Nx, Ny, Ne; array_type=CuArray, T=T_prec)
    stats = EnsembleStatistics(Nx, Ny; array_type=CuArray, T=T_prec)
else
    T_prec = Float64
    grid = Grid(Nx, Ny; array_type=Array, T=T_prec)
    params = Parameters(; dt = 0.005, T_end = 0.05, T=T_prec) # very short run (10 steps)
    state = ModelState(Nx, Ny, Ne; array_type=Array, T=T_prec)
    stats = EnsembleStatistics(Nx, Ny; array_type=Array, T=T_prec)
end
psolver = PoissonSolver(state.omega, grid)

# 2. Setup IC
init_initial_condition!(state.omega, grid, params)

# 3. Step
solve_spde!(state, grid, params, stats, psolver)

println("Test passed!")
