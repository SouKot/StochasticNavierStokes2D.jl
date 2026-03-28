module IO

using Printf
using HDF5
import Statistics

export compute_statistics!, save_statistics

function compute_statistics!(stats, state)
    omega = state.omega
    Nx, Ny, Ne = size(omega)
    T = eltype(omega)
    
    # 1. Zero out original stats buffers
    stats.mean .= zero(T)
    stats.variance .= zero(T)
    stats.moment3 .= zero(T)
    stats.moment4 .= zero(T)
    
    # Pass 1: Compute Raw Power Sums exactly. (S1, S2, S3, S4)
    # This enables us to mathematically merge fully independent batches seamlessly.
    for i in 1:Ne
        @views @. stats.mean += omega[:, :, i]
        @views @. stats.variance += omega[:, :, i]^2
        @views @. stats.moment3 += omega[:, :, i]^3
        @views @. stats.moment4 += omega[:, :, i]^4
    end
    
    # Normalize by the chunk size
    stats.mean ./= T(Ne)
    stats.variance ./= T(Ne)
    stats.moment3 ./= T(Ne)
    stats.moment4 ./= T(Ne)

    return nothing
end

function save_statistics(filename::String, state, stats, t::Float64)
    # Save the 2D fields to HDF5
    h5open(filename, "cw") do file
        # Check if group exists
        grp_name = @sprintf("t_%0.4f", t)
        if haskey(file, grp_name)
            delete_object(file, grp_name)
        end
        grp = create_group(file, grp_name)
        
        # Bring arrays back to CPU if they are on GPU for HDF5 writing
        # Save Power Sums mapping
        write(grp, "S1", Array(stats.mean))
        write(grp, "S2", Array(stats.variance))
        write(grp, "S3", Array(stats.moment3))
        write(grp, "S4", Array(stats.moment4))
        
        # Save raw realizations for offline POD/DO/ADO evaluation
        write(grp, "omega", Array(state.omega))
        write(grp, "psi", Array(state.psi))
    end
end

end # module
