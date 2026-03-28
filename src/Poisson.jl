module Poisson

using FFTW

export PoissonSolver, solve_poisson!, apply_implicit_diffusion!

struct PoissonSolver{F, iF, A<:AbstractArray, Ac<:AbstractArray}
    plan::F
    iplan::iF
    K2::A      # Wavenumber squared array sizes (Nx, Ny)
    Kx::A      # Wavenumber x
    Ky::A      # Wavenumber y
    dealias_filter::A # 2/3 Rule mask
    buffer_c::Ac # Reusable complex buffer for the batch
    batch_size::Int
    n_batches::Int
    Ne::Int
end

function PoissonSolver(omega::AbstractArray{T, 3}, grid) where T
    Nx, Ny, Ne = size(omega)
    Lx, Ly = grid.Lx, grid.Ly
    
    # Configure FFTW to use all available Julia threads
    FFTW.set_num_threads(Threads.nthreads())
    
    # Dynamic Sub-batching: Calculate safe batch size based on memory footprint
    # The Spectral solver requires ~7 complex arrays (14 real arrays) simultaneously per batch element.
    # We restrict the maximum instantaneous complex buffer allocation to roughly 1.0 GB to safely 
    # fit inside an 8GB GPU while leaving room for the main simulation state fields.
    bytes_per_slice = Nx * Ny * 14 * sizeof(T)
    target_max_bytes = 1024^3 # 1 GB
    
    max_safe_batch = min(Ne, max(1, floor(Int, target_max_bytes / bytes_per_slice)))
    
    # Find the largest possible batch size <= max_safe_batch that perfectly divides Ne
    batch_size = max_safe_batch
    while batch_size > 1 && Ne % batch_size != 0
        batch_size -= 1
    end
    n_batches = div(Ne, batch_size)
    
    # Allocate a reusable complex buffer of size (Nx, Ny, batch_size)
    buffer_c = similar(omega, Complex{T}, Nx, Ny, batch_size)
    
    # We plan the in-place FFT over the first two dimensions of the buffer
    P = plan_fft!(buffer_c, (1, 2))
    iP = plan_ifft!(buffer_c, (1, 2))
    
    # Pre-calculate wavenumber matrix
    kx = fftfreq(Nx, Nx/Lx) .* (2 * T(π))
    ky = fftfreq(Ny, Ny/Ly) .* (2 * T(π))
    
    kmax_x = maximum(abs.(kx))
    kmax_y = maximum(abs.(ky))

    K2_host = zeros(T, Nx, Ny)
    Kx_host = zeros(T, Nx, Ny)
    Ky_host = zeros(T, Nx, Ny)
    dealias_host = ones(T, Nx, Ny)
    
    for j in 1:Ny
        for i in 1:Nx
            K2_host[i, j] = kx[i]^2 + ky[j]^2
            Kx_host[i, j] = kx[i]
            Ky_host[i, j] = ky[j]
            
            # Orszag 2/3 Rule de-aliasing mask
            if abs(kx[i]) >= (2.0/3.0) * kmax_x || abs(ky[j]) >= (2.0/3.0) * kmax_y
                dealias_host[i, j] = 0.0
            end
        end
    end
    K2_host[1, 1] = T(1.0) # Avoid div by zero
    
    K2 = copyto!(similar(omega, Nx, Ny), K2_host)
    Kx = copyto!(similar(omega, Nx, Ny), Kx_host)
    Ky = copyto!(similar(omega, Nx, Ny), Ky_host)
    dealias_filter = copyto!(similar(omega, Nx, Ny), dealias_host)
    
    return PoissonSolver(P, iP, K2, Kx, Ky, dealias_filter, buffer_c, batch_size, n_batches, Ne)
end

function get_batch_range(solver::PoissonSolver, b::Int)
    start_idx = (b - 1) * solver.batch_size + 1
    end_idx = b * solver.batch_size
    return start_idx:end_idx
end

function solve_poisson!(psi::A1, omega::A2, solver::PoissonSolver) where {A1, A2}
    for b in 1:solver.n_batches
        idx = get_batch_range(solver, b)
        
        # 1. Copy to complex buffer
        solver.buffer_c .= complex.(view(omega, :, :, idx))
        
        # 2. In-place FFT
        solver.plan * solver.buffer_c
        
        # 3. Apply physics (solve Poisson)
        solver.buffer_c ./= solver.K2
        @views solver.buffer_c[1, 1, :] .= 0
        
        # 4. In-place Inverse FFT
        solver.iplan * solver.buffer_c
        
        # 5. Extract Real part
        view(psi, :, :, idx) .= real.(solver.buffer_c)
    end
    return nothing
end

function apply_implicit_diffusion!(omega::A, solver::PoissonSolver, factor) where {A}
    for b in 1:solver.n_batches
        idx = get_batch_range(solver, b)
        
        # 1. Copy to complex buffer
        solver.buffer_c .= complex.(view(omega, :, :, idx))
        
        # 2. In-place FFT
        solver.plan * solver.buffer_c
        
        # 3. Apply Implicit Diffusion factor
        solver.buffer_c ./= (1.0 .+ factor .* solver.K2)
        @views solver.buffer_c[1, 1, :] .= 0
        
        # 4. In-place Inverse FFT
        solver.iplan * solver.buffer_c
        
        # 5. Extract Real part and overwrite original omega state
        view(omega, :, :, idx) .= real.(solver.buffer_c)
    end
    return nothing
end

end # module
