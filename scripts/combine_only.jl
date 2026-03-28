include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D
using .StochasticNavierStokes2D.Visualization
using Printf
using HDF5

function combine_batches(final_filename, batch_filenames, Nx, Ny, N_batches, Ne_batch)
    println("Combining $(length(batch_filenames)) batches into $final_filename ...")
    
    # First, find all common keys (time steps)
    keys_list = h5open(batch_filenames[1], "r") do file
        keys(file)
    end
    
    Ne_total = N_batches * Ne_batch

    h5open(final_filename, "cw") do f_out
        for k in keys_list
            if haskey(f_out, k)
                delete_object(f_out, k)
            end
            grp = create_group(f_out, k)
            
            # Pass 1: compute true global mean from the batches
            global_mean = zeros(Float64, Nx, Ny)
            for b_file in batch_filenames
                h5open(b_file, "r") do f_in
                    global_mean .+= read(f_in[k]["S1"])
                end
            end
            global_mean ./= N_batches
            
            # Preallocate global moment accumulators (use Float64 inside the loop for precision!)
            # These MUST be reset to zero for every unique time-step `k`!
            var_global = zeros(Float64, Nx, Ny)
            m3_global = zeros(Float64, Nx, Ny)
            m4_global = zeros(Float64, Nx, Ny)
            
            # Create HDF5 datasets for the massive raw fields directly
            # We use Float32 to match the GPU arrays and save 50% disk space
            ds_omega = create_dataset(grp, "omega", datatype(Float32), dataspace(Nx, Ny, Ne_total), chunk=(Nx, Ny, 100))
            ds_psi   = create_dataset(grp, "psi", datatype(Float32), dataspace(Nx, Ny, Ne_total), chunk=(Nx, Ny, 100))
            
            offset = 1
            for b_file in batch_filenames
                h5open(b_file, "r") do f_in
                    # Read the raw 3D array for this batch natively into CPU RAM (~800 MB)
                    omega_b = read(f_in[k]["omega"])
                    psi_b   = read(f_in[k]["psi"])
                    
                    # Compute true central differences perfectly out-of-core
                    for i in 1:Ne_batch
                        diff = Float64.(omega_b[:, :, i]) .- global_mean
                        var_global .+= diff.^2
                        m3_global  .+= diff.^3
                        m4_global  .+= diff.^4
                    end
                    
                    # Stream the raw arrays into the final global HDF5 file
                    ds_omega[:, :, offset:(offset + Ne_batch - 1)] = omega_b
                    ds_psi[:, :, offset:(offset + Ne_batch - 1)]   = psi_b
                end
                offset += Ne_batch
            end
            
            # Normalize the central moments
            var_global ./= Ne_total
            m3_global  ./= Ne_total
            m4_global  ./= Ne_total
            
            # Extract standard Excess Kurtosis exactly as defined in the paper:
            # kurtosis_excess = (m4 / Ne) - 3 * (variance)^2
            m4_excess = m4_global .- 3.0 .* (var_global.^2)
            
            # Save our perfectly computed global statistics
            write(grp, "mean", Float32.(global_mean))
            write(grp, "variance", Float32.(var_global))
            write(grp, "moment3", Float32.(m3_global))
            write(grp, "moment4", Float32.(m4_excess))
        end
    end
end

function main()
    Nx, Ny = 256, 256
    Ne_total = 12800   
    N_batches = 4      
    Ne_batch = div(Ne_total, N_batches)
    
    T_end = 1.0     
    dt = 0.000625   
    scheme = :Spectral

    println("\n>>> Independent Batch Combiner Tool <<<")
    println("Targeting Scheme:  ", scheme)
    println("Total Ensembles:   ", Ne_total)
    println("Reconstructing from ", N_batches, " batches of ", Ne_batch, " realizations...")
    println("--------------------------------\n")

    batch_filenames = [
        "batch_1_statistics_Spectral_N3200_Nx256.h5",
        "batch_2_statistics_Spectral_N3200_Nx256.h5",
        "batch_3_statistics_Spectral_N3200_Nx256.h5",
        "batch_4_statistics_Spectral_N3200_Nx256.h5"
    ]
    
    # Verify files exist
    for f in batch_filenames
        if !isfile(f)
            error("Missing batch file: $f. Cannot combine!")
        end
    end

    # 1. Combine Power Sums into exact global Central Moments
    final_filename = "/media/sourabh/Data/2D-Navier_Stokes-results/statistics_$(scheme)_N$(Ne_total)_Nx$(Nx)_dt$(dt)_T$(T_end).h5"
    combine_batches(final_filename, batch_filenames, Nx, Ny, N_batches, Ne_batch)
    
    # 2. Cleanup orphaned chunks
    println("Merge successful! Deleting raw uncombined batch files...")
    for f in batch_filenames
        rm(f, force=true)
    end
    
    # 3. Generate Visualizations Native
    println("\nGenerating Visualization PNGs from newly combined Global Statistics...")
    
    try
        keys_list = h5open(final_filename, "r") do file
            keys(file)
        end
        println("Found global checkpoints: ", keys_list)

        last_t = 0.0
        for k in keys_list
            t_str = replace(k, "t_" => "")
            t_val = parse(Float64, t_str)
            last_t = max(last_t, t_val)
            try
                Visualization.plot_statistics(final_filename, t_val)
            catch e
                println("Skipping $k due to output error: $e")
            end
        end

        # Generate 1D slices at the absolute latest checkpoint
        println("Generating 1D slice plots for Global ensemble at t=$last_t ...")
        try
            Visualization.plot_statistics_1d_slice(final_filename, last_t, 0.125, Nx, Ny)
            Visualization.plot_statistics_1d_slice(final_filename, last_t, 0.875, Nx, Ny)
        catch e
            println("Skipping 1D slices due to error: $e")
        end
    catch e
        println("Failed to read $final_filename to plot: $e")
    end

    println("Complete! Standalone merge and plot successfully resolved.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
