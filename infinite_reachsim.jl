#infinite_reachsim
using Interpolations
using Statistics
using ColorSchemes
using MAT
using Plots

module reaching_et_data

export surf_v_t


function plot_vt_surfaces(; simfile="data/out50_fill.mat", f=14002, cf=4.2, az=27, el=33.6,plotstyle="GLMakie")
"""
function plot_vt_surfaces(; simfile="data/out50_fill.mat", f=14002, cf=4.2, az=27, el=33.6,plotstyle="GLMakie")
  
    Plot the surfaces of distance, duration, and peak speed as functions of time valuation and distance.

    Parameters
    ----------
    simfile : str
        Path to the MAT file containing the simulation results.
    f : int
        Figure number for the plot.
    cf : float
        Conversion factor for time valuation.
    az : float
        Azimuth angle for the 3D plot.
    el : float
        Elevation angle for the 3D plot.
    plotstyle : str
        Plotting style: "Plots" or "GLMakie".

    Returns
    -------
    fig : Figure
        The generated figure object.
"""
  # Handle default parameters:

  vclps = [az, el]
      
  # Create a custom colorscheme approximating the MATLAB one
  # Colors in RGB format (0-1)
  cs = range(RGB(239/255, 237/255, 245/255), RGB(117/255, 107/255, 177/255), length=100)
  
  # Load MAT file

  data = matread(simfile)
  distance = data["distance"]
  duration = data["duration"]
  peakspeed = data["peakspeed"]
  timeValuation = data["timeValuation"]
  # traj = data["traj"]
  

  filled_xs = copy(timeValuation)
  filled_ys = copy(distance)
  filled_vals = copy(peakspeed)

  # Replace zeros with NaN
  filled_xs[filled_xs .== 0.0] .= NaN
  filled_ys[filled_ys .== 0.0] .= NaN

  (ct, distance, duration)  = fillmissingsurf_gridded(timeValuation, distance, duration)
  (_, _, peakspeed)         = fillmissingsurf_gridded(timeValuation, distance, peakspeed)
  
  # Add zero columns
  distance  = hcat(zeros(size(distance, 1)), distance)
  duration  = hcat(zeros(size(distance, 1)), duration)
  peakspeed = hcat(zeros(size(distance, 1)), peakspeed)
  ct = hcat(ct[:, 1], ct)
  
  # Convert ct to ct_mech
  eff_mech = 0.25
  ct_mech = ct * eff_mech

  # Plotting
  if plotstyle == "Plots"
      
    # Create a new figure
    fig = plot(layout=(2, 3), size=(575, 350), background_color=:white)
    
    # First subplot (equivalent to subplot(231) in MATLAB)
    subplot = plot!(fig[1], ct_mech, distance, duration, 
        st=:surface, 
        color=cs, 
        alpha=0.5, 
        showaxis=true, 
        xlabel="Time valuation Cₜ (W)", 
        ylabel="Distance L (m)", 
        zlabel="Duration T (s)",
        xlim=(0, 50/cf),
        ylim=(0, 0.55),
        zlim=(0, 2.5),
        camera=(vclps[1], vclps[2])
    )
    
    # Second subplot (equivalent to subplot(234) in MATLAB)
    subplot = plot!(fig[4], ct_mech, distance, peakspeed, 
        st=:surface, 
        color=cs, 
        alpha=0.5, 
        showaxis=true, 
        xlabel="Cₜ (W)", 
        ylabel="Distance L (m)", 
        zlabel="Peak speed V (m/s)",
        xlim=(0, 50/cf),
        ylim=(0, 0.55),
        zlim=(0, 1),
        camera=(vclps[1], vclps[2])
    )
    
    display(fig)
    return fig
  elseif plotstyle == "GLMakie"
    return interactive_surface(ct_mech, distance, duration, peakspeed, az=az, el=el, cf=cf)
  else
    error("Invalid plotstyle: $plotstyle")
  end
end

function fillmissingsurf_gridded(xs, ys, vals)

  """
Interpolate missing values in a 2D surface using gridded interpolation.
"""
# Check dimensions
if size(xs) != size(ys) || size(xs) != size(vals)
    error("Input arrays must have the same dimensions")
end

# Create copies to avoid modifying originals
filled_xs = copy(xs)
filled_ys = copy(ys)
filled_vals = copy(vals)

# Replace zeros with NaN
filled_xs[filled_xs .== 0.0] .= NaN
filled_ys[filled_ys .== 0.0] .= NaN

# Find indices with valid data (no NaNs)
valid_mask = .!isnan.(filled_xs) .&& .!isnan.(filled_ys) .&& .!isnan.(filled_vals)
valid_indices = findall(valid_mask)

if length(valid_indices) == 0
    error("No valid data points for interpolation")
end

# Extract coordinates and values for valid points
valid_points = [(filled_xs[i], filled_ys[i]) for i in valid_indices]
valid_xs = [p[1] for p in valid_points]
valid_ys = [p[2] for p in valid_points]
valid_values = [filled_vals[i] for i in valid_indices]

# Find indices that need interpolation
nan_indices = findall(.!valid_mask)
unique_xs = sort(unique(valid_xs))
unique_ys = sort(unique(valid_ys))
    
# If no interpolation needed, return original
if length(nan_indices) == 0
    return filled_xs, filled_ys, filled_vals
end

# Rebuild the values array to match the grid structure
gridded_values = Array{Float64}(undef, length(unique_xs), length(unique_ys))
fill!(gridded_values, NaN)

# Fill in the known values
for i in 1:length(valid_xs)
    x_idx = findfirst(==(valid_xs[i]), unique_xs)
    y_idx = findfirst(==(valid_ys[i]), unique_ys)
    gridded_values[x_idx, y_idx] = valid_values[i]
end

# Create the interpolation object
itp = interpolate((unique_xs, unique_ys), gridded_values, Gridded(Linear()))
    
# Interpolate missing values
for idx in nan_indices
    # For points with missing x or y, we need to estimate them first
    # This is a simplified approach - actual implementation depends on data structure
    i, j = idx.I  # Get row and column indices
    
    # Estimate x coordinate if needed
    if isnan(filled_xs[idx])
        # Use column average or nearby values
        col_values = filled_xs[:, j]
        valid_col = col_values[.!isnan.(col_values)]
        if length(valid_col) > 0
            filled_xs[idx] = mean(valid_col)
        else
            # Fallback to global mean if no valid points in column
            filled_xs[idx] = mean(valid_xs)
        end
    end
    
    # Estimate y coordinate if needed
    if isnan(filled_ys[idx])
        # Use row average or nearby values
        row_values = filled_ys[i, :]
        valid_row = row_values[.!isnan.(row_values)]
        if length(valid_row) > 0
            filled_ys[idx] = mean(valid_row)
        else
            # Fallback to global mean if no valid points in row
            filled_ys[idx] = mean(valid_ys)
        end
    end
    
    # Now interpolate the value
    try
        filled_vals[idx] = itp(filled_xs[idx], filled_ys[idx])
    catch
        # For points outside the convex hull of the data
        # find nearest valid point
        distances = [sqrt((filled_xs[idx] - x)^2 + (filled_ys[idx] - y)^2) 
                    for (x, y) in zip(valid_xs, valid_ys)]
        nearest_idx = argmin(distances)
        filled_vals[idx] = valid_values[nearest_idx]
    end
end

return filled_xs, filled_ys, filled_vals
end # /function fillmissingsurf_gridded

# Option: Using GLMakie for interactive visualization
function interactive_surface(ct_mech, distance, duration, peakspeed; az=27, el=33.6, cf=4.2)
  using GLMakie
    
    # Create a new figure
    fig = Figure(resolution=(1000, 700), fontsize=16)
    
    # Add a 3D axis for the duration surface
    ax1 = Axis3(fig[1, 1], 
               xlabel="Time valuation Cₜ (W)",
               ylabel="Distance L (m)",
               zlabel="Duration T (s)",
               title="Duration Surface")
    
    # Create the surface
    surf1 = surface!(ax1, ct_mech, distance, duration,
               colormap=:plasma,
               transparency=true,
               alpha=0.85,
               shading=true)
    
    # Set limits
    xlims!(ax1, 0, 50/cf)
    ylims!(ax1, 0, 0.55)
    zlims!(ax1, 0, 2.5)
    
    # Add colorbar
    Colorbar(fig[1, 2], surf1, label="Duration (s)")
    
    # Add a 3D axis for the peak speed surface
    ax2 = Axis3(fig[2, 1], 
               xlabel="Time valuation Cₜ (W)",
               ylabel="Distance L (m)",
               zlabel="Peak Speed V (m/s)",
               title="Peak Speed Surface")
    
    # Create the surface
    surf2 = surface!(ax2, ct_mech, distance, peakspeed,
               colormap=:plasma,
               transparency=true,
               alpha=0.85,
               shading=true)
    
    # Set limits
    xlims!(ax2, 0, 50/cf)
    ylims!(ax2, 0, 0.55)
    zlims!(ax2, 0, 1.0)
    
    # Add colorbar
    Colorbar(fig[2, 2], surf2, label="Peak Speed (m/s)")
    
    # Initial camera positions
    rotate_cam!(ax1, deg2rad(el), deg2rad(az), 0)
    rotate_cam!(ax2, deg2rad(el), deg2rad(az), 0)
    
    # Add camera control sliders
    azimuth_slider = Slider(fig[3, 1], range=0:1:360, startvalue=az)
    elevation_slider = Slider(fig[4, 1], range=0:1:90, startvalue=el)
    
    # Labels for sliders
    Label(fig[3, 1, Top()], "Azimuth")
    Label(fig[4, 1, Top()], "Elevation")
    
    # Connect sliders to camera rotation
    on(azimuth_slider.value) do val
        rotate_cam!(ax1, deg2rad(elevation_slider.value[]), deg2rad(val), 0)
        rotate_cam!(ax2, deg2rad(elevation_slider.value[]), deg2rad(val), 0)
    end
    
    on(elevation_slider.value) do val
        rotate_cam!(ax1, deg2rad(val), deg2rad(azimuth_slider.value[]), 0)
        rotate_cam!(ax2, deg2rad(val), deg2rad(azimuth_slider.value[]), 0)
    end
    
    return fig
end
end
