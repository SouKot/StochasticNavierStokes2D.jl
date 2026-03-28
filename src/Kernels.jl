module Kernels

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    eval(:( @init_parallel_stencil(CUDA, Float32, 2) ))
else
    eval(:( @init_parallel_stencil(Threads, Float64, 2) ))
end

export compute_derivatives!, compute_laplacian_and_jacobian!, compute_explicit_rhs_cn!

@parallel_indices (ix, iy, ie) function _compute_derivatives!(u, v, psi, dx, dy)
    # u = d_psi / d_y
    # v = - d_psi / d_x
    # Using central differences with periodic boundary conditions
    # We will assume ghost nodes are manually handled or we just use modulo arithmetic
    
    Nx, Ny, _ = size(psi)
    
    # Periodic indices
    ix_p = ix < Nx ? ix + 1 : 1
    ix_m = ix > 1  ? ix - 1 : Nx
    iy_p = iy < Ny ? iy + 1 : 1
    iy_m = iy > 1  ? iy - 1 : Ny
    
    dpsi_dx = (psi[ix_p, iy, ie] - psi[ix_m, iy, ie]) / (2.0 * dx)
    dpsi_dy = (psi[ix, iy_p, ie] - psi[ix, iy_m, ie]) / (2.0 * dy)
    
    u[ix, iy, ie] = dpsi_dy
    v[ix, iy, ie] = -dpsi_dx
    
    return nothing
end

function compute_derivatives!(u, v, psi, grid)
    @parallel (1:grid.Nx, 1:grid.Ny, 1:size(psi, 3)) _compute_derivatives!(u, v, psi, grid.dx, grid.dy)
    return nothing
end

@parallel_indices (ix, iy, ie) function _compute_laplacian_and_jacobian!(rhs, omega, psi, u, v, mu, dx, dy)
    Nx, Ny, _ = size(omega)

    ix_p = ix < Nx ? ix + 1 : 1
    ix_m = ix > 1  ? ix - 1 : Nx
    iy_p = iy < Ny ? iy + 1 : 1
    iy_m = iy > 1  ? iy - 1 : Ny

    # Laplacian of omega
    d2omega_dx2 = (omega[ix_p, iy, ie] - 2.0*omega[ix, iy, ie] + omega[ix_m, iy, ie]) / (dx^2)
    d2omega_dy2 = (omega[ix, iy_p, ie] - 2.0*omega[ix, iy, ie] + omega[ix, iy_m, ie]) / (dy^2)
    laplacian_omega = d2omega_dx2 + d2omega_dy2

    # Jacobian (advection term): J(psi, omega) = d_psi/dx * d_omega/dy - d_psi/dy * d_omega/dx
    # Alternative: u * d_omega/dx + v * d_omega/dy
    domega_dx = (omega[ix_p, iy, ie] - omega[ix_m, iy, ie]) / (2.0 * dx)
    domega_dy = (omega[ix, iy_p, ie] - omega[ix, iy_m, ie]) / (2.0 * dy)
    
    advection = u[ix, iy, ie] * domega_dx + v[ix, iy, ie] * domega_dy

    # Eq: d_omega/dt = - advection + mu * laplacian
    rhs[ix, iy, ie] = -advection + mu * laplacian_omega

    return nothing
end

function compute_laplacian_and_jacobian!(rhs, omega, psi, u, v, grid, params)
    @parallel (1:grid.Nx, 1:grid.Ny, 1:size(omega, 3)) _compute_laplacian_and_jacobian!(rhs, omega, psi, u, v, params.mu, grid.dx, grid.dy)
    return nothing
end

@parallel_indices (ix, iy, ie) function _compute_explicit_rhs_cn!(rhs, omega, u, v, mu, dx, dy)
    Nx, Ny, _ = size(omega)

    ix_p = ix < Nx ? ix + 1 : 1
    ix_m = ix > 1  ? ix - 1 : Nx
    iy_p = iy < Ny ? iy + 1 : 1
    iy_m = iy > 1  ? iy - 1 : Ny

    # Laplacian of omega
    d2omega_dx2 = (omega[ix_p, iy, ie] - 2.0*omega[ix, iy, ie] + omega[ix_m, iy, ie]) / (dx^2)
    d2omega_dy2 = (omega[ix, iy_p, ie] - 2.0*omega[ix, iy, ie] + omega[ix, iy_m, ie]) / (dy^2)
    laplacian_omega = d2omega_dx2 + d2omega_dy2

    # Jacobian (advection term)
    domega_dx = (omega[ix_p, iy, ie] - omega[ix_m, iy, ie]) / (2.0 * dx)
    domega_dy = (omega[ix, iy_p, ie] - omega[ix, iy_m, ie]) / (2.0 * dy)
    
    advection = u[ix, iy, ie] * domega_dx + v[ix, iy, ie] * domega_dy

    # CN explicit rhs
    rhs[ix, iy, ie] = -advection + (mu / 2.0) * laplacian_omega

    return nothing
end

function compute_explicit_rhs_cn!(rhs, omega, u, v, grid, params)
    @parallel (1:grid.Nx, 1:grid.Ny, 1:size(omega, 3)) _compute_explicit_rhs_cn!(rhs, omega, u, v, params.mu, grid.dx, grid.dy)
    return nothing
end

end # module
