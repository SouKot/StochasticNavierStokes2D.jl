include("../src/StochasticNavierStokes2D.jl")
using .StochasticNavierStokes2D.Visualization: plot_statistics
using CairoMakie: save

println("Plotting statistics at t=0.05...")
fig = plot_statistics("statistics_N6400.h5", 1.0)
save("moments_plot.png", fig)
println("Plot saved to moments_plot.png")
