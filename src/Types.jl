module Types

export Grid, Parameters, ModelState, EnsembleStatistics

struct Grid{T<:AbstractFloat, V<:AbstractVector{T}}
    Nx::Int
    Ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    x::V
    y::V
end

function Grid(Nx, Ny, Lx=1.0, Ly=1.0; array_type=Array, T=Float64)
    dx = Lx / Nx
    dy = Ly / Ny
    x = collect(range(0, Lx, length=Nx+1)[1:end-1])
    y = collect(range(0, Ly, length=Ny+1)[1:end-1])
    x_arr = array_type(T.(x))
    y_arr = array_type(T.(y))
    return Grid{T, typeof(x_arr)}(Nx, Ny, T(Lx), T(Ly), T(dx), T(dy), x_arr, y_arr)
end

struct Parameters{T<:AbstractFloat}
    mu::T        # Viscosity (0.0002)
    dt::T        # Time step (0.0025)
    T_end::T     # Final time (1.0)
    N_steps::Int
    epsilon::T   # Perturbation amplitude (0.3)
    delta::T     # Shear layer width (0.025)
end

function Parameters(; mu=0.0002, dt=0.0025, T_end=1.0, epsilon=0.3, delta=0.025, T=Float64)
    N_steps = round(Int, T_end / dt)
    return Parameters{T}(T(mu), T(dt), T(T_end), N_steps, T(epsilon), T(delta))
end

struct ModelState{A<:AbstractArray}
    omega::A   # Vorticity (Nx, Ny, Ne)
    psi::A     # Streamfunction (Nx, Ny, Ne)
    u::A       # x-velocity (Nx, Ny, Ne)
    v::A       # y-velocity (Nx, Ny, Ne)
    rhs::A     # Right-hand side (Nx, Ny, Ne)
end

function ModelState(Nx, Ny, N_ensemble; array_type=Array, T=Float64)
    # The CUDA.jl pooling allocator geometrically doubles buffer sizes if requests 
    # happen too fast. A single monolithic 4.0GB 4D array request triggers an 
    # impossible 15.6 GB pool reservation. We instantiate the arrays one-by-one 
    # and explicitly sync the garbage collector/memory pool between them.
    omega = fill!(array_type{T, 3}(undef, Nx, Ny, N_ensemble), zero(T))
    GC.gc(true)
    psi = fill!(array_type{T, 3}(undef, Nx, Ny, N_ensemble), zero(T))
    GC.gc(true)
    u = fill!(array_type{T, 3}(undef, Nx, Ny, N_ensemble), zero(T))
    GC.gc(true)
    v   = fill!(array_type{T, 3}(undef, Nx, Ny, N_ensemble), zero(T))
    GC.gc(true)
    rhs = fill!(array_type{T, 3}(undef, Nx, Ny, N_ensemble), zero(T))
    
    return ModelState{typeof(omega)}(omega, psi, u, v, rhs)
end

struct EnsembleStatistics{A<:AbstractArray}
    mean::A      # (Nx, Ny)
    variance::A  # (Nx, Ny)
    moment3::A   # (Nx, Ny)
    moment4::A   # (Nx, Ny)
end

function EnsembleStatistics(Nx, Ny; array_type=Array, T=Float64)
    mean = fill!(array_type{T, 2}(undef, Nx, Ny), zero(T))
    variance = fill!(array_type{T, 2}(undef, Nx, Ny), zero(T))
    moment3 = fill!(array_type{T, 2}(undef, Nx, Ny), zero(T))
    moment4 = fill!(array_type{T, 2}(undef, Nx, Ny), zero(T))
    return EnsembleStatistics{typeof(mean)}(mean, variance, moment3, moment4)
end

end # module
