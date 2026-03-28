using HDF5

filename = "statistics_N6400.h5"

h5open(filename, "r") do file
    keys_list = keys(file)
    last_k = keys_list[end]
    grp = file[last_k]
    mean = read(grp, "mean")
    
    # 0 to 1 with 128 points
    Ny = 128
    y_coords = collect(range(0, 1.0, length=Ny+1)[1:end-1])
    
    # Check x=0.125 (index 17)
    idx_125 = 17
    mean_slice = mean[idx_125, :]
    
    peaks = []
    for i in 2:Ny-1
        if mean_slice[i] > mean_slice[i-1] && mean_slice[i] > mean_slice[i+1] && mean_slice[i] > 0
            push!(peaks, (y_coords[i], mean_slice[i]))
        end
    end
    
    println("Peaks at x=0.125 for 1.0s:")
    for p in peaks
        println("  y = $(round(p[1], digits=4)), value = $(round(p[2], digits=4))")
    end
    
    # Check x=0.875 (index 113)
    idx_875 = 113
    mean_slice_875 = mean[idx_875, :]
    
    peaks_875 = []
    for i in 2:Ny-1
        if mean_slice_875[i] > mean_slice_875[i-1] && mean_slice_875[i] > mean_slice_875[i+1] && mean_slice_875[i] > 0
            push!(peaks_875, (y_coords[i], mean_slice_875[i]))
        end
    end
    
    println("\nPeaks at x=0.875 for 1.0s:")
    for p in peaks_875
        println("  y = $(round(p[1], digits=4)), value = $(round(p[2], digits=4))")
    end
end
