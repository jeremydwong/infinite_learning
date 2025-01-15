# 3rd order eye model
print("Warning! wip!")
sleep(5)
using InfiniteOpt, Ipopt, Plots, Distributions, ControlSystems
include("plotpvu.jl")

# function run3rdorder(;tcs=[0.224, 0.013, 0.010], c_fr  = .05, d = .01, c_t=1)
## Third order eye model Harris & Wolpert
# three time constants, 224, 13, 10 ms
# (Need 3rd order to get bell shape)
# Transfer function, convert to continuous-time state space
tcs=[0.224, 0.013, 0.010]
c_fr  = .01
d = .005
c_t=20
tfeye = tf(1, [tcs[1], 1])*tf(1, [tcs[2], 1])*tf(1, [tcs[3], 1]) # Harris & Wolpert 3rd-order eye model
eyeden = tfeye.matrix[1].den  # denominator coefficients in ascending order: b[0] + b[1]*s + b[2]*s^2
eyesdn = InfiniteModel(Ipopt.Optimizer)
# time
@infinite_parameter(eyesdn, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))
# scaling of time
@variable(eyesdn, 0.001 <= t_f <= 10, start = 1) # scaling of time by t_f.

# position(p), velocity(v), acceleration(a), control(u)
@variable(eyesdn, p, Infinite(t), start = (t)-> 1*t)
@variable(eyesdn, v, Infinite(t))
@variable(eyesdn, a, Infinite(t))
@variable(eyesdn, u, Infinite(t))
@variable(eyesdn, e_damp, Infinite(t))
@variable(eyesdn, e_k, Infinite(t))
@variable(eyesdn, f_tot, Infinite(t))
@variable(eyesdn, e_tot, Infinite(t))

# slack variables for force rate
@variable(eyesdn, fddotp >= 0, Infinite(t))
@variable(eyesdn, fddotm <= 0, Infinite(t))

# dynamics constraints connecting [p, v, a] to [v, a, u] 
@constraint(eyesdn, ∂(p,t) == t_f * v) # Dynamics
@constraint(eyesdn, ∂(v,t) == t_f * a)
@constraint(eyesdn, ∂(a,t) == t_f * (-eyeden[0]/eyeden[3]*p - eyeden[1]/eyeden[3]*v - eyeden[2]/eyeden[3]*a + 1/eyeden[3]*u) ) 

# u is fdot; dudt is fdotdot.
@constraint(eyesdn, ∂(u,t) == t_f * (fddotp + fddotm) )

# define as constraints equations for damping and kinetic energy
@constraint(eyesdn, e_damp == eyeden[1]/eyeden[2]*v*v)
@constraint(eyesdn, e_k == 1/2*1*v*v)
k1 = eyeden[0]/eyeden[3]
k2 = eyeden[1]/eyeden[3]
k3 = eyeden[2]/eyeden[3]
k4 = 1/eyeden[3]
@constraint(eyesdn, f_tot == (-k1*p - k2*v + k4*u - ∂(a,t)) /k_3 )
@constraint(eyesdn, e_tot == integral(f_tot*v,t)*t_f)
#normalized mass; 1. I rearrange equation for xddd to get a normalized-mass solution.
# k2/k3 ends up being the damping coefficient.
@objective(eyesdn, Min, integral(eyeden[1]/eyeden[2]*v*v + c_fr * (fddotp - fddotm),t) + c_t * t_f) # positive work and force rate 

# Boundary conditions
@constraint(eyesdn, p(0) == 0) # Boundary conditions
@constraint(eyesdn, p(1) == d)
@constraint(eyesdn, v(0) == 0)
@constraint(eyesdn, v(1) == 0)
@constraint(eyesdn, a(0) == 0)
@constraint(eyesdn, a(1) == 0)
@constraint(eyesdn, u(0) == 0)
@constraint(eyesdn, u(1) == 0)
@constraint(eyesdn, fddotp(0) == 0)
@constraint(eyesdn, fddotp(1) == 0)
@constraint(eyesdn, fddotm(0) == 0)
@constraint(eyesdn, fddotm(1) == 0)

optimize!(eyesdn)
t_ = supports(t)
traces = (p,v,u,fddotp,fddotm)
pfig=plotpvu(t, traces..., title="min e+t",t_scaled = value(t_f))
pfig
#   return t,traces,eyesdn,pfig, value(t_f), value(e_damp), value(e_k), value(f_tot)
# end
  
pl = plot()
plot!(pl, t_, [value(e_damp),value(e_k),value(f_tot).*value(v)], label=["damping" "kinetic" "mech"], xlabel="t", ylabel="energy")



(t, traces, eyesdn,pfig,t_f, e_d,e_k,f_tot) = run3rdorder()

# bell-shapes for 3rd order: d = [0.01,0.1], c_fr = [0.005,0.001] 
# (t, traces, eyesdn,pfig) = run3rdorder(d=0.01,c_fr=.001)

# find saturation?...
ds = [0.01,0.02,0.03,0.04,0.1,1,10]
pf = plot(layout = length(traces))
for d in ds
  (t, traces, eyesdn,pfig,tfinal) = run3rdorder(d=d,c_fr=.00005)
  
  tfinal
  plotpvu!(pf, t, traces...; t_scaled = tfinal)
end
pf
