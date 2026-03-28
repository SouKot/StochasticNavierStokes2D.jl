module Visualization

using CairoMakie
using HDF5
using Printf

export plot_statistics

function plot_statistics(filename::String, t::Float64)
    grp_name = @sprintf("t_%0.4f", t)
    
    h5open(filename, "r") do file
        if !haskey(file, grp_name)
            error("Time $t not found in $filename")
        end
        
        grp = file[grp_name]
        mean = read(grp, "mean")
        variance = read(grp, "variance")
        mom3 = read(grp, "moment3")
        mom4 = read(grp, "moment4")
        
        fig = Figure(size=(1000, 800))
        
        ax1 = Axis(fig[1, 1], title="Mean")
        hm1 = heatmap!(ax1, mean, colormap=:jet)
        Colorbar(fig[1, 2], hm1)
        
        ax2 = Axis(fig[1, 3], title="Variance")
        hm2 = heatmap!(ax2, variance, colormap=:jet)
        Colorbar(fig[1, 4], hm2)
        
        ax3 = Axis(fig[2, 1], title="3rd Moment")
        hm3 = heatmap!(ax3, mom3, colormap=:jet)
        Colorbar(fig[2, 2], hm3)
        
        ax4 = Axis(fig[2, 3], title="Excess 4th Moment")
        hm4 = heatmap!(ax4, mom4, colormap=:jet)
        Colorbar(fig[2, 4], hm4)
        
        for ext in [".png", ".svg", ".eps"]
            out_name = replace(filename, ".h5" => "") * "_$(grp_name)$(ext)"
            save(out_name, fig)
        end
        println("Saved visualization to: ", replace(filename, ".h5" => "") * "_$(grp_name).*")
        
        return fig
    end
end

function plot_statistics_1d_slice(filename::String, t::Float64, x_val::Float64, Nx::Int=128, Ny::Int=128)
    grp_name = @sprintf("t_%0.4f", t)
    
    h5open(filename, "r") do file
        if !haskey(file, grp_name)
            error("Time $t not found in $filename")
        end
        
        grp = file[grp_name]
        mean = read(grp, "mean")
        variance = read(grp, "variance")
        mom3 = read(grp, "moment3")
        mom4 = read(grp, "moment4")
        
        x_idx = round(Int, x_val * Nx) + 1
        x_idx = clamp(x_idx, 1, Nx)
        
        y_coords = collect(range(0, 1.0, length=Ny+1)[1:end-1])
        
        fig = Figure(size=(800, 800))
        
        # Invert mean and 3rd moment for plotting to match author's positive visual convention
        ax1 = Axis(fig[1, 1], title="mean", xlabel="y", ylabel="Eω")
        lines!(ax1, y_coords, -mean[x_idx, :], label="ref", linestyle=:dot)
        axislegend(ax1, position=:lt)
        
        ax2 = Axis(fig[1, 2], title="variance", xlabel="y", ylabel="Eω²")
        lines!(ax2, y_coords, variance[x_idx, :], label="ref", linestyle=:dot)
        
        ax3 = Axis(fig[2, 1], title="3rd moment", xlabel="y", ylabel="Eω³")
        lines!(ax3, y_coords, -mom3[x_idx, :], label="ref", linestyle=:dot)
        
        ax4 = Axis(fig[2, 2], title="excess 4th moment", xlabel="y", ylabel="Eω⁴ - 3(Eω²)²")
        lines!(ax4, y_coords, mom4[x_idx, :], label="ref", linestyle=:dot)
        
        for ext in [".png", ".svg", ".eps"]
            out_name = replace(filename, ".h5" => "") * "_$(grp_name)_x_$(x_val)$(ext)"
            save(out_name, fig)
        end
        println("Saved 1D slice visualization to: ", replace(filename, ".h5" => "") * "_$(grp_name)_x_$(x_val).*")
        
        return fig
    end
end

end # module
