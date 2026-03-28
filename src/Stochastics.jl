module Stochastics

using Random

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    eval(:( @init_parallel_stencil(CUDA, Float32, 2) ))
else
    eval(:( @init_parallel_stencil(Threads, Float64, 2) ))
end

export init_forcing_amplitude!, init_initial_condition!

# As derived from the paper Section 4.2
# sigma = (-0.1*pi*cos(2*pi*x)*cos(2*pi*y), 0.1*pi*sin(2*pi*x)*sin(2*pi*y))
# We just need to define the scalar amplitude since it hits the vorticity equation

@parallel_indices (ix, iy) function _init_forcing_amplitude!(sigma_amp, x, y, T)
    pi = T(π)
    # Note: paper defines vector \sigma for velocity. 
    # For vorticity, \nabla x \sigma gives a scalar forcing. 
    # Since only \omega has explicit stochastic evolution per eq 39:
    # d\omega = ... + \sigma dW. 
    # Wait, the paper says "sigma" is the spatially dependent random forcing.
    # We will assume a scalar field for the amplitude on omega directly.
    # The paper says: \sigma = (-0.1\pi \cos(2\pi x)\cos(2\pi y), 0.1\pi \sin(2\pi x)\sin(2\pi y))
    # It seems to inject it as a vector into the momentum equations implicitly?
    # No, eq (39) explicitly has \sigma dW in the \omega equation. 
    # If \sigma is a vector but in the \omega eq it's a scalar, it means it's the curl of a vector forcing.
    # Curl of that \sigma vector = \partial \sigma_y / \partial x - \partial \sigma_x / \partial y
    # = \partial(0.1\pi \sin(2\pi x)\sin(2\pi y))/\partial x - \partial(-0.1\pi \cos(2\pi x)\cos(2\pi y))/\partial y
    # = 0.2\pi^2 \cos(2\pi x)\sin(2\pi y) - 0.2\pi^2 \cos(2\pi x)\sin(2\pi y) = 0?
    # Let's carefully re-read the paper's eq 39. 
    # "b(x,y) = \gamma \exp..." Wait, that is Section 4.3.
    # Section 4.2: \sigma = (-0.1\pi \cos(2\pi x)\cos(2\pi y), 0.1\pi \sin(2\pi x)\sin(2\pi y))
    # And eq 39: \partial \omega ... = \mu \Delta \omega dt + \sigma dW
    # It must be that the \sigma vector is what is provided, and we take its curl for the \omega forcing, OR \sigma in eq 39 is just a scalar field. Let's provide a generic scalar field.
    
    # I will construct a scalar field based on taking a norm or using one component.
    # Actually, if we look at the paper text exactly: "sigma = (...)"
    # Let me use the curl of the provided vector to be safe.
    # curl(sig_x, sig_y) = dsig_y/dx - dsig_x/dy
    # = 0.2 * pi^2 * cos(2pi x) sin(2pi y) - (0.2 * pi^2 * cos(2pi x) sin(2pi y)) = 0.
    # Wait! If it's a vector Wiener process W = (W1, W2)^T, then:
    # d\omega = sigma_x dW_1 + sigma_y dW_2 ?
    # Eq 39: \partial \omega ... = ... \sigma dW
    # So it's an inner product: sigma \cdot dW.
    # Yes! \sigma is a vector, dW is a vector. 
    # So \sigma \cdot dW = sigma_x dW_1 + sigma_y dW_2.
    
    sigma_x = -T(0.1) * pi * cos(T(2) * pi * x[ix]) * cos(T(2) * pi * y[iy])
    sigma_y =  T(0.1) * pi * sin(T(2) * pi * x[ix]) * sin(T(2) * pi * y[iy])
    
    sigma_amp[ix, iy, 1] = sigma_x
    sigma_amp[ix, iy, 2] = sigma_y
    return nothing
end

function init_forcing_amplitude!(sigma_amp, grid)
    @parallel (1:grid.Nx, 1:grid.Ny) _init_forcing_amplitude!(sigma_amp, grid.x, grid.y, eltype(grid.x))
    return nothing
end

@parallel_indices (ix, iy, ie) function _init_initial_condition!(omega, x, y, delta, epsilon, C, T)
    pi = T(π)
    
    # I(x) = 1 + epsilon * (cos(4*pi*x) - 1)
    Ix = T(1.0) + epsilon * (cos(T(4)*pi*x[ix]) - T(1.0))
    
    # exp_term = exp(- I(x) * (y - 0.5)^2 / (2 * delta^2))
    exp_term = exp(-Ix * (y[iy] - T(0.5))^2 / (T(2) * delta^2))
    
    # omega = C - 1/(2*delta) * exp_term (Restored negative sign per original text)
    val = C - (T(1.0) / (T(2)*delta)) * exp_term
    
    omega[ix, iy, ie] = val
    return nothing
end

function init_initial_condition!(omega, grid, params, C=0.0)
    # We will need to compute C such that the integral over the domain is 0.
    # First we compute without C.
    T = eltype(grid.x)
    @parallel (1:grid.Nx, 1:grid.Ny, 1:size(omega, 3)) _init_initial_condition!(omega, grid.x, grid.y, params.delta, params.epsilon, T(0.0), T)
    
    # Calculate integral to find C
    # integral = sum(omega[:,:,1]) * dx * dy
    # To make integral 0, C = - integral / (Lx * Ly) ? No.
    # If omega_new = omega_old + C, then integral_new = integral_old + C * Lx * Ly
    # So C = - integral_old / (Lx * Ly)
    
    # Only need to compute on one realization since they are all the same
    omega_host = Array(omega[:, :, 1])
    integral_old = sum(omega_host) * grid.dx * grid.dy
    C_corr = -integral_old / (grid.Lx * grid.Ly)
    
    @parallel (1:grid.Nx, 1:grid.Ny, 1:size(omega, 3)) _init_initial_condition!(omega, grid.x, grid.y, params.delta, params.epsilon, T(C_corr), T)
    return nothing
end

end # module
