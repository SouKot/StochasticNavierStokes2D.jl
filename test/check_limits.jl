using HDF5
using Printf

filename = "statistics_N6400.h5"

println("Scanning exact min/max bounds for all 4 moments in $filename at t=1.000...")
try
    h5open(filename, "r") do file
        # Grab the last time key
        keys_list = keys(file)
        last_k = keys_list[end]
        
        grp = file[last_k]
        mean = read(grp, "mean")
        var = read(grp, "variance")
        mom3 = read(grp, "moment3")
        mom4 = read(grp, "moment4")
        
        println("\n---- OUR SIMULATION RANGES ($last_k) ----")
        @printf("1) Mean:          [%7.3f, %7.3f]\n", minimum(mean), maximum(mean))
        @printf("2) Variance:      [%7.3f, %7.3f]\n", minimum(var), maximum(var))
        @printf("3) 3rd Moment:    [%7.3f, %7.3f]\n", minimum(mom3), maximum(mom3))
        @printf("4) Excess 4th:    [%7.3f, %7.3f]\n", minimum(mom4), maximum(mom4))
        println("----------------------------------------")
    end
catch e
    println("Error: $e")
end
