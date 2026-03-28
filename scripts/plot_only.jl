include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D.Visualization
using HDF5

function main()
    println(">>> Visualization Generation Tool <<<")
    # You can change this target to any specific HDF5 file you want to plot.
    target_file = "/media/sourabh/Data/2D-Navier_Stokes-results/statistics_Spectral_N12800_Nx256_dt0.000625_T1.0.h5"
    
    if !isfile(target_file)
        println("Error: Could not find target file '$target_file' in the directory.")
        println("Please edit scripts/plot_only.jl and update 'target_file' to your desired .h5 file.")
        return
    end

    println("Loading HDF5 File: ", target_file)

    try
        keys_list = h5open(target_file, "r") do file
            keys(file)
        end
        println("Found checkpoints: ", keys_list)

        last_t = 0.0
        for k in keys_list
            t_str = replace(k, "t_" => "")
            t_val = parse(Float64, t_str)
            last_t = max(last_t, t_val)
            try
                Visualization.plot_statistics(target_file, t_val)
            catch e
                println("Skipping 2D plot for $k due to output error: $e")
            end
        end

        # Generate 1D slices at the absolute latest checkpoint for x=0.125 and x=0.875
        println("Generating 1D slice plots for ensemble at t=$last_t ...")
        Nx, Ny = 256, 256
        try
            Visualization.plot_statistics_1d_slice(target_file, last_t, 0.125, Nx, Ny)
            Visualization.plot_statistics_1d_slice(target_file, last_t, 0.875, Nx, Ny)
        catch e
            println("Skipping 1D slices due to error: $e")
        end
        
        println("\nSuccess! Vector graphics (.png, .svg, .eps) arrays generated natively.")
    catch e
        println("Failed to read $target_file to plot: $e")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
