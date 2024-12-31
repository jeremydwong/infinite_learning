# utility for plotting subplots of position vs time, velocity vs time,
# and actuation vs. time. For infiniteopt variables.

using Plots
import InfiniteOpt.InfiniteModel, InfiniteOpt.GeneralVariableRef


"""
plotpvu!(phandle, t::GeneralVariableRef, traces::GeneralVariableRef...) 

   In-place plotter for optimal trajectories.
   Makes subplots for a variable number of InfiniteOpt variables, which are of
   type `GeneralVariableRef`. Optional keyword arguments include strings `title` and
   `ylabels` array of strings to label y axes. Returns a plot handle.
   The name `plotpvu` comes from plotting position, velocity, and control u, but
   any InfiniteOpt variables can be plotted.

   Note: Deprecating plotpvu in favor of multiplot.
"""
function plotpvu!(pfig::Plots.Plot, t::GeneralVariableRef, traces::GeneralVariableRef...; 
    title="", ylabels=fill("", (length(traces,))))
    ts = supports(t)
    for i in eachindex(traces)
        y = value(traces[i], ndarray=true)
        plot!(pfig[i], legend=false, xlabel="t", ylabel=ylabels[i])
        if size(y,2) > 1 # plot a single trace, or a mean + multiple traces
            plot!(pfig[i], ts, y, linewidth = 0.2)
            plot!(pfig[i], ts, mean(y, dims=2), linewidth=2)
        else
            plot!(pfig[i], ts, y) # thinner lines if multiple
        end
    end
    plot!(pfig[1], title=title) # title to first subplot
end

"""
`phandle = plotpvu(t::GeneralVariableRef, trace1::GeneralVariableRef...)`

   Plots subplots for an InfiniteOpt model m, given time t, and a variable number of traces.
   Will initialize a 2-column layout and return the plot handle. Optional keyword arguments
   include a `title` string, and an array `ylabel` of strings for labeling y-axes.
   
   Note: Deprecating plotpvu in favor of multiplot.
"""
function plotpvu(t::GeneralVariableRef, traces::GeneralVariableRef...; 
    title="", ylabels=fill("", (length(traces,))))
    pfig = plot(layout=length(traces))
    plotpvu!(pfig, t, traces...; title=title, ylabels=ylabels)
    return pfig
end

# My version of a "huber loss" absolute value. It's smooth and differentiable
# and allows for different costs for positive and negative work. 
# Use smaller eps to get a sharper corner. eps=1 is roughly okay.
# use negfactor to make the negative leg cost different from positive
# by that proportional amount.
function softabs(x; eps=1e-4, negfactor = 1.)
    if x >= eps
        return x
    elseif x >= -eps
        return (negfactor+1)/(4*eps)*x^2 + (1-negfactor)/2*x + (1/4*eps*(1+negfactor))
    else
        return -negfactor*x
    end
end # habs function

# hard relu
relu(x) = x*(x>=0)

# soft relu, set c higher to get sharper corner
softrelu(x, c=1.) = log(1+exp(c*x))/c

#plot(x->softabs(x,eps=1e0,negfactor=1))

"""
    supportsvector(x::GeneralVariableRef) returns a vector of the supports from `InfiniteOpt` variable `x`. 
"""
supportsvector(x::GeneralVariableRef) = vcat(collect.(supports(x))...)

"""
    multiplot(a, ...; names) produces multiple plots of `InfiniteOpt` variables. One or more 
    such variables (of type `InfiniteOpt.GeneralVariableRef`) are plotted as separate
    subplots, each with their associated x-axis. A vector of strings `names` is used to
    label the y-axes of the plots.
"""
multiplot # a function defined by the plot recipe below

@userplot MultiPlot # produces a function multiplot

@recipe function f(mp::MultiPlot; names = nothing, title="")
    layout := length(mp.args)
    legend := false
    title := ""
    xlabel := "t"

    for (i,p) in enumerate(mp.args)
        @series begin
            subplot := i
            if i == 1
                title := title
            end
            if names != nothing
                ylabel := names[i]
            end
            (vcat(collect.(supports(p))...), value(p,ndarray=true))
        end
    end
end

