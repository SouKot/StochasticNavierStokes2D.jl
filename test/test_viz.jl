include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D.Visualization
using HDF5

filename = "statistics_N10.h5"
println("Testing 1D visualization plotting on $filename...")

try
    # Read the available keys dynamically since t=0.0 isn't saved by default
    keys_list = h5open(filename, "r") do file
        keys(file)
    end
    println("Available time checkpoints in file: ", keys_list)
    
    if isempty(keys_list)
        println("File is empty!")
    else
        # Grab the last time checkpoint like "t_0.0500" and strip the "t_" to parse the float
        last_t_str = replace(keys_list[end], "t_" => "")
        last_t = parse(Float64, last_t_str)
        
        println("Testing plotting for t = $last_t ...")
        Visualization.plot_statistics_1d_slice(filename, last_t, 0.125, 128, 128)
        Visualization.plot_statistics_1d_slice(filename, last_t, 0.14, 128, 128)
        println("Test passed successfully! Images were written.")
    end
catch e
    println("Test failed: ", e)
end
