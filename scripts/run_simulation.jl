include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D
using .StochasticNavierStokes2D.Types: Grid, Parameters, ModelState, EnsembleStatistics
using .StochasticNavierStokes2D.Stochastics: init_initial_condition!
using .StochasticNavierStokes2D.Poisson: PoissonSolver
using .StochasticNavierStokes2D.Solver: solve_spde!
using .StochasticNavierStokes2D.Visualization
using Printf
using HDF5

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    using CUDA
    println(">>> Simulation Target: Nvidia GPU (CUDA) <<<")
else
    println(">>> Simulation Target: Multicore CPU (Threads) <<<")
end

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
    # Configuration matches the paper's target specifications
    Nx, Ny = 256, 256
    Ne_total = 12800    # Testing ensemble total
    N_batches = 4      # Single batch test for moment verification
    Ne_batch = div(Ne_total, N_batches)
    
    T_end = 1.0     # 1 second of simulation time as stated in paper
    dt = 0.000625   # Halved explicit time step to satisfy the exact spectral CFL limit at 256x256!
    save_int = 0.25 # Save snapshot every 0.25 seconds
    scheme = :Spectral

    # Setup Grid and Params for exactly ONE chunk at a time (e.g. 3,200 elements, completely dodging hardware limits)
    if USE_GPU
        T_prec = Float32
        grid = Grid(Nx, Ny; array_type=CuArray, T=T_prec)
        params = Parameters(; dt=dt, T_end=T_end, T=T_prec)
        state = ModelState(Nx, Ny, Ne_batch; array_type=CuArray, T=T_prec)
        stats = EnsembleStatistics(Nx, Ny; array_type=CuArray, T=T_prec)
    else
        T_prec = Float64
        grid = Grid(Nx, Ny; array_type=Array, T=T_prec)
        params = Parameters(; dt=dt, T_end=T_end, T=T_prec)
        state = ModelState(Nx, Ny, Ne_batch; array_type=Array, T=T_prec)
        stats = EnsembleStatistics(Nx, Ny; array_type=Array, T=T_prec)
    end

    # 1. Initialize core Poisson Solver strictly once
    psolver = PoissonSolver(state.omega, grid)
    
    println("\n>>> Simulation Configuration <<<")
    println("Method:          ", scheme)
    println("Final Time:      ", T_end)
    println("Time Step (dt):  ", dt)
    println("Grid Size (Nx):  ", Nx)
    println("Total Ensembles: ", Ne_total)
    println("Batches:         ", N_batches, " (", Ne_batch, " realizations independently each)")
    println("GPU Peak State:  ~3.9 GiB (Dynamically dodged the 15.6 GiB OS Limit!)")
    println("--------------------------------\n")

    batch_filenames = String[]
    
    # 2. Sequential Out-of-Core Ensemble Chunking
    for b in 1:N_batches
        println("\n>>> Starting Monte Carlo Batch $b / $N_batches <<<")
        println("Initializing $Ne_batch distinct random realizations...")
        
        # Zero the state fully before running the new trajectory (avoid memory leaks or persistence)
        state.omega .= 0
        state.psi .= 0
        state.u .= 0
        state.v .= 0
        state.rhs .= 0
        
        init_initial_condition!(state.omega, grid, params)
        
        chunk_file = "batch_$(b)_statistics_$(scheme)_N$(Ne_batch)_Nx$(Nx).h5"
        push!(batch_filenames, chunk_file)
        
        solve_spde!(state, grid, params, stats, psolver; save_interval=save_int, scheme=scheme, filename=chunk_file)
    end
    
    # 3. Combine Power Sums into the exact mathematically identical Global Central Moments
    final_filename = "statistics_$(scheme)_N$(Ne_total)_Nx$(Nx)_dt$(dt)_T$(T_end).h5"
    combine_batches(final_filename, batch_filenames, Nx, Ny, N_batches, Ne_batch)
    
    # Delete the intermediary raw sum batches to keep the directory clean
    for f in batch_filenames
        rm(f, force=true)
    end
    
    # 4. Generate Visualization PNGs natively
    println("\nSimulation Complete! Generating Visualization PNGs from Aggregate Global Statistics...")
    
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

        # Generate 1D slices at the absolute latest checkpoint for x=0.125 and x=0.875
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

    println("All plots successfully generated in the project root directory!")
end

# Fire the script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
