module Solver

using Printf
using Random: randn!

using ..Types
using ..Stochastics
using ..Poisson
using ..Kernels
using ..SpectralKernels
using ..IO

export solve_spde!

function solve_spde!(state::ModelState, grid::Grid, params::Parameters, stats::EnsembleStatistics, solver::PoissonSolver; save_interval=0.1, scheme=:Euler, filename="statistics.h5")
    
    T = eltype(grid.x)
    dt = params.dt
    mu = params.mu
    Ne = size(state.omega, 3)
    
    # Pre-allocate random forcing amplitude
    sigma_amp = similar(state.omega, grid.Nx, grid.Ny, 2)
    init_forcing_amplitude!(sigma_amp, grid)
    
    # Pre-allocate Spectral Workspace if requested
    spectral_space = (scheme == :Spectral) ? SpectralWorkspace(solver) : nothing
    
    # Pre-allocate Weiner increments for the vector Wiener process
    # The paper explicitly states W = (W1, W2)^T. Meaning it's a 2-dimensional stochastic process, 
    # NOT a full spatial white noise field. Thus, we only need 2 random scalars per realization.
    dW1 = similar(state.omega, 1, 1, Ne)
    dW2 = similar(state.omega, 1, 1, Ne)
    
    t = 0.0
    step = 0
    next_save_time = 0.0
    
    println("Starting Simulation...")
    
    while step < params.N_steps
        
        # 1. Solve Poisson for Streamfunction (psi)
        solve_poisson!(state.psi, state.omega, solver)
        
        # 2. Compute Velocities (u, v)
        compute_derivatives!(state.u, state.v, state.psi, grid)
        
        if scheme == :Euler
            # 3. Compute deterministic RHS (Advection + Diffusion)
            compute_laplacian_and_jacobian!(state.rhs, state.omega, state.psi, state.u, state.v, grid, params)
            
            # 4. Generate Stochastic Forcing
            # dW1, dW2 ~ N(0, dt)
            randn!(dW1)
            randn!(dW2)
            dW1 .*= sqrt(dt)
            dW2 .*= sqrt(dt)
            
            # 5. Time update (Euler-Maruyama)
            # omega_new = omega_old + rhs * dt + \sigma_1 dW1 + \sigma_2 dW2
            @views @. state.omega += state.rhs * dt + sigma_amp[:,:,1] * dW1 + sigma_amp[:,:,2] * dW2

        elseif scheme == :CrankNicolson
            # 3. Compute Explicit half-step RHS
            compute_explicit_rhs_cn!(state.rhs, state.omega, state.u, state.v, grid, params)

            # 4. Generate Stochastic Forcing
            randn!(dW1)
            randn!(dW2)
            dW1 .*= sqrt(dt)
            dW2 .*= sqrt(dt)

            # 5. Construct Explicit side of equation F
            @views @. state.omega += state.rhs * dt + sigma_amp[:,:,1] * dW1 + sigma_amp[:,:,2] * dW2

            # 6. Apply Implicit Fourier Solve using memory-safe batching
            apply_implicit_diffusion!(state.omega, solver, (dt * params.mu / 2.0))
            
        elseif scheme == :Spectral
            # 3. Compute Exact explicit RHS Pseudo-Spectrally
            compute_spectral_explicit_rhs!(state.rhs, state.omega, state.psi, state.u, state.v, spectral_space, solver)
            
            # 4. Generate Stochastic Forcing
            randn!(dW1)
            randn!(dW2)
            dW1 .*= sqrt(dt)
            dW2 .*= sqrt(dt)

            # 5. Construct Explicit side of equation F
            @views @. state.omega += state.rhs * dt + sigma_amp[:,:,1] * dW1 + sigma_amp[:,:,2] * dW2

            # 6. Apply Implicit Fourier Solve
            # The Pseudo-Spectral explicit step contains pure advection (no explicit diffusion half).
            # Therefore, we MUST apply the full Backward Euler diffusion weight here to conserve exact energy.
            apply_implicit_diffusion!(state.omega, solver, (dt * params.mu))
        else
            error("Unknown scheme: $scheme")
        end
        
        t += dt
        step += 1
        
        # Checkpoint / Statistics
        if t >= next_save_time || step == params.N_steps
            @printf("Step %d / %d, Time: %.3f\n", step, params.N_steps, t)
            compute_statistics!(stats, state)
            
            # Export to the specifically defined HDF5 chunk file
            save_statistics(filename, state, stats, t)
            
            next_save_time += save_interval
        end
    end
    
    println("Simulation Complete!")
end

end # module
