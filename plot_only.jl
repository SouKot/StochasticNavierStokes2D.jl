include("src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D.Visualization
using HDF5

filename = "statistics_N6400.h5"
println("Attempting to dynamically read $filename and generate visualizations...")

try
    keys_list = h5open(filename, "r") do file
        keys(file)
    end
    println("Found checkpoints: ", keys_list)
    
    last_t = 0.0
    for k in keys_list
        t_str = replace(k, "t_" => "")
        t_val = parse(Float64, t_str)
        last_t = max(last_t, t_val)
        try
            Visualization.plot_statistics(filename, t_val)
        catch e
            println("Skipping $k due to output error: $e")
        end
    end
    
    println("Generating 1D slice plots for t=$last_t at x=0.125 and x=0.14 ...")
    try
        Visualization.plot_statistics_1d_slice(filename, last_t, 0.125, 128, 128)
        Visualization.plot_statistics_1d_slice(filename, last_t, 0.14, 128, 128)
    catch e
        println("Skipping 1D slices due to error: $e")
    end
    
catch e
    println("Failed to read $filename to plot: $e")
end

println("Done!")
