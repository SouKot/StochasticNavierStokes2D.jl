module SpectralKernels

using FFTW
using ..Types
using ..Poisson

export SpectralWorkspace, compute_spectral_explicit_rhs!

struct SpectralWorkspace{Ac<:AbstractArray}
    buffer_w_hat::Ac
    buffer_u_hat::Ac
    buffer_v_hat::Ac
    buffer_wx_hat::Ac
    buffer_wy_hat::Ac
    buffer_psi_hat::Ac
end

function SpectralWorkspace(solver::PoissonSolver)
    buffer_w_hat = similar(solver.buffer_c)
    buffer_u_hat = similar(solver.buffer_c)
    buffer_v_hat = similar(solver.buffer_c)
    buffer_wx_hat = similar(solver.buffer_c)
    buffer_wy_hat = similar(solver.buffer_c)
    buffer_psi_hat = similar(solver.buffer_c)
    return SpectralWorkspace(buffer_w_hat, buffer_u_hat, buffer_v_hat, buffer_wx_hat, buffer_wy_hat, buffer_psi_hat)
end

function get_batch_range(solver::PoissonSolver, b::Int)
    start_idx = (b - 1) * solver.batch_size + 1
    end_idx = b * solver.batch_size
    return start_idx:end_idx
end

function compute_spectral_explicit_rhs!(rhs, omega, psi, u, v, workspace::SpectralWorkspace, solver::PoissonSolver)
    
    for b in 1:solver.n_batches
        idx = get_batch_range(solver, b)
        
        # 1. Transform omega to frequency space
        workspace.buffer_w_hat .= complex.(view(omega, :, :, idx))
        solver.plan * workspace.buffer_w_hat
        
        # 2. Compute frequency derivatives exactly, completely de-aliased
        @views @. workspace.buffer_psi_hat = workspace.buffer_w_hat / solver.K2
        @views workspace.buffer_psi_hat[1, 1, :] .= 0 # Enforce mean 0 streamfunction
        
        @views @. workspace.buffer_u_hat  = workspace.buffer_psi_hat * (im * solver.Ky) * solver.dealias_filter
        @views @. workspace.buffer_v_hat  = workspace.buffer_psi_hat * (-im * solver.Kx) * solver.dealias_filter
        @views @. workspace.buffer_wx_hat = workspace.buffer_w_hat * (im * solver.Kx) * solver.dealias_filter
        @views @. workspace.buffer_wy_hat = workspace.buffer_w_hat * (im * solver.Ky) * solver.dealias_filter
        
        # Implicitly zero the mean modes
        @views workspace.buffer_u_hat[1, 1, :] .= 0
        @views workspace.buffer_v_hat[1, 1, :] .= 0
        @views workspace.buffer_wx_hat[1, 1, :] .= 0
        @views workspace.buffer_wy_hat[1, 1, :] .= 0

        # 3. IFFT directly in place
        solver.iplan * workspace.buffer_u_hat
        solver.iplan * workspace.buffer_v_hat
        solver.iplan * workspace.buffer_wx_hat
        solver.iplan * workspace.buffer_wy_hat
        solver.iplan * workspace.buffer_psi_hat
        
        # 4. Compute explicit RHS in physical space: (u * dx_w + v * dy_w)
        @views @. u[:, :, idx] = real(workspace.buffer_u_hat)
        @views @. v[:, :, idx] = real(workspace.buffer_v_hat)
        @views @. psi[:, :, idx] = real(workspace.buffer_psi_hat)
        
        @views @. rhs[:, :, idx] = -(real(workspace.buffer_u_hat) * real(workspace.buffer_wx_hat) + real(workspace.buffer_v_hat) * real(workspace.buffer_wy_hat))
    end
end

end # module
