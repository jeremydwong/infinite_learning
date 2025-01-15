# 2rd order eye model


using InfiniteOpt, Ipopt, Plots, Distributions, ControlSystems
include("plotpvu.jl")

function trapz(t, y)
    return sum((y[1:end-1] + y[2:end]) .* diff(t))/2
end
function cumtrapz(t, y)
    return cumsum((y[1:end-1] + y[2:end]) .* diff(t))/2
end

## 2nd order eye model Harris & Wolpert
# two time constants, 224, 13 ms
# (Need 3rd order to get bell shape)
c_t = 20
c_fr = 0.01
D = 0.01

# Transfer function, convert to continuous-time state space
tfeye = tf(1, [0.224, 1])*tf(1, [0.013, 1]) # Harris & Wolpert 3rd-order eye model
eyeden = tfeye.matrix[1].den  # denominator coefficients in ascending order: b[0] + b[1]*s + b[2]*s^2

eye2nd = InfiniteModel(Ipopt.Optimizer)
# time
@infinite_parameter(eye2nd, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))
# scaling of time
@variable(eye2nd, 0.001 <= t_f <= 50, start = 1) 

# position(p), velocity(v), acceleration(a), control(u)
@variable(eye2nd, p, Infinite(t), start = (t)-> 1*t)
@variable(eye2nd, v, Infinite(t))
@variable(eye2nd, u, Infinite(t))
@variable(eye2nd, e_mech, Infinite(t))
@variable(eye2nd, e_damp, Infinite(t))
@variable(eye2nd, e_k, Infinite(t))
@finite_parameter(eye2nd, δ == D)
@finite_parameter(eye2nd, cᵣ == c_fr)

# dynamics constraints connecting [p, v, a] to [v, a, u] 
@constraint(eye2nd, ∂(p,t) == t_f * v) 
k1 = eyeden[0]/eyeden[2]
k2 = eyeden[1]/eyeden[2]
@constraint(eye2nd, ∂(v,t) == t_f * (k1*(u-p) - k2*v) ) 
# A = [0 1; -k1 -k2] # position and velocity
# B = [0 k1]'
@variable(eye2nd, fdot, Infinite(t))
@constraint(eye2nd, ∂(u,t) == t_f * fdot )

# slack variables for force rate
@variable(eye2nd, fddotp >= 0, Infinite(t))
@variable(eye2nd, fddotm <= 0, Infinite(t))
@constraint(eye2nd, ∂(fdot,t) == t_f * (fddotp + fddotm) )

#note 2025-01-15: integrating by dimensionless time t; so multiply by t_f.
@objective(eye2nd, Min, integral((k1*(u-p))*v,t)*t_f + cᵣ*integral(fddotp,t)*t_f + c_t * t_f) # positive work and force rate 

# Boundary conditions
@constraint(eye2nd, p(0) == 0.0) # Boundary conditions
@constraint(eye2nd, p(1) == δ)
@constraint(eye2nd, v(0) == 0.0)
@constraint(eye2nd, v(1) == 0.0)
@constraint(eye2nd, u(0) == 0.0)
@constraint(eye2nd, u(1) == 0.0)
@constraint(eye2nd, fdot(0) == 0.0)
@constraint(eye2nd, fdot(1) == 0.0)
@constraint(eye2nd, fddotp(0) == 0.0)
@constraint(eye2nd, fddotp(1) == 0.0)
@constraint(eye2nd, fddotm(0) == 0.0)
@constraint(eye2nd, fddotm(1) == 0.0)

#prevent the optimizer from running too long
set_optimizer_attribute(eye2nd, "max_cpu_time", 60.)
set_optimizer_attributes(eye2nd, "tol" => 1e-4, "max_iter" => 500)

# run once
set_value(δ, 0.01)
optimize!(eye2nd)
v_ = value(v)
u_ = value(u)
p_ = value(p)
t_ = supports(t)*value(t_f)
t_f_ = value(t_f)
p_joint = (k1*(u_-p_).*v_)
e_joint = [[0];cumtrapz(t_, p_joint)]
e_damp  = [[0];cumtrapz(t_,k2*v_.*v_)]
e_k     = 1/2*1*v_.*v_
pl_reach = plot()
pl_reach = plot(layout = (3,2),size = (800,800))

# run across range of distances to show satuating peak velocity.
δs = [0.001,0.002,0.003,0.004,0.005, 0.01, 0.02,0.03,0.04,0.05,0.06]
# init vpeaks
vpeaks = zeros(length(δs))
for (i,δ_) in enumerate(δs)
    set_value(δ, δ_)
    optimize!(eye2nd)
    v_ = value(v)
    u_ = value(u)
    p_ = value(p)
    t_ = supports(t)*value(t_f)
    t_f_ = value(t_f)
    p_joint = (k1*(u_-p_).*v_)
    e_joint = [[0];cumtrapz(t_, p_joint)]
    e_damp  = [[0];cumtrapz(t_,k2*v_.*v_)]
    e_k     = 1/2*1*v_.*v_
    plot!(t_,p_,title="position",subplot=1,legend = false)
    plot!(t_,v_,title="velocity",subplot=2,legend = false)
    plot!(t_,u_,title="control",subplot=3,legend = false)
    if i == 1
      plot!(t_, [e_joint, e_damp, e_k,e_damp+e_k], label=["joint" "damp" "kinetic" "damp+kinetic"], xlabel="t", ylabel="energy",subplot=4)    
    end
    vpeaks[i] = maximum(v_)
end
scatter!(δs, vpeaks,subplot=5,legend = false,ylimits=(0,0.6),xlimits=(0,0.07))
plot!(δs, vpeaks,subplot=5,legend = false,ylimits=(0,0.6),xlimits=(0,0.07),xlabel="Distance",ylabel="Peak V")
pl_reach