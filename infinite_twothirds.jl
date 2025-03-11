# Point mass, energy and time.

using InfiniteOpt, Ipopt, Plots, Distributions
include("plotpvu.jl")

## Understanding simple E + FR + T model

## Work and force-rate and time with
# point mass model
# function pointmassmodel(;c_fr = 0.05, c_t = 1,k_b = 5)
model = InfiniteModel(Ipopt.Optimizer)
c_t = 1
gain_fr = 1/4
c_fr = 0.05*gain_fr
k_b = 1
@infinite_parameter(model, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))
@variable(model, p, Infinite(t), start = (t)->1*t)
@variable(model, v, Infinite(t))
@variable(model, fp >=0, Infinite(t))
@variable(model, fn <=0, Infinite(t))
@variable(model, fdot, Infinite(t))
@finite_parameter(model, δ == 1) # fixed distance
@finite_parameter(model, τ == 1) # fixed time

@variable(model, fddotp >= 0, Infinite(t))
@variable(model, fddotm <= 0, Infinite(t))

@variable(model, 0.001 <= t_f <= 10, start = 1) # scaling of time by tf.

@constraint(model, ∂(p,t)        == t_f * v) # Dynamics
@constraint(model, ∂(v,t)        == t_f * ((fp + fn) - k_b*v))
@constraint(model, (∂(fp+fn,t))  == t_f * fdot) 
@constraint(model, ∂(fdot, t)    == t_f * (fddotp + fddotm))

@objective(model, Min, integral(fp*v - fn*v + c_fr*(fddotp - fddotm), t) + c_t * t_f) # abs work and force rate 

@constraint(model, p(0) == 0) # Boundary conditions
@constraint(model, p(1) == δ)
@constraint(model, v(0) == 0)
@constraint(model, v(1) == 0)
@constraint(model, fp(0)+fn(0) == 0)
@constraint(model, fp(1)+fn(1) == 0)
@constraint(model, fdot(0) == 0)
@constraint(model, fdot(1) == 0)
@constraint(model, t_f == τ)

set_optimizer_attribute(model, "max_cpu_time", 60.)
set_optimizer_attributes(model, "tol" => 1e-4, "max_iter" => 500)

optimize!(model)
t_val = supports(t)*value(t_f)

  # end
function cumtrapz(t, y)
  return cumsum((y[1:end-1] + y[2:end]) .* diff(t))/2
end

function packageresults()
  v_ = value(v)
  p_ = value(p)
  t_ = supports(t)*value(t_f)
  f_ = value(fp) + value(fn)
  p_mech = (f_.*v_)
  e_met  = [[0];cumtrapz(t_, 1*(p_mech .> 0) .* p_mech)] + (-2) .* [[0];cumtrapz(t_, (p_mech .< 0) .* p_mech)]
  e_mech = [[0];cumtrapz(t_, p_mech)]
  e_damp = [[0];cumtrapz(t_,k_b*v_.*v_)]
  e_k    = 1/2*1*v_.*v_

  fddp_  = value(fddotp)
  fddm_  = value(fddotm)
  c_fr_  = value(c_fr)
  e_fr   = [[0]; c_fr_ * cumtrapz(t_,fddp_ - fddm_)]
  return t_, p_, v_, f_, e_mech, e_damp, e_k, e_fr,e_met
end
#amp scales with freq to to the
f_base = 1
fs = [.8, 1.2, 1.6,2, 2.6,3.5] *f_base # freq for a half-cycle. A to B. 
Ps = 1 ./ fs

f_loop = plot(layout = (4,2))
# set size of f_loop
plot!(size = (800,800))
vpeak_vec = zeros(length(Ps))
e_fr_vec  = zeros(length(Ps))
e_w_vec   = zeros(length(Ps))
e_met_vec = zeros(length(Ps))
δ_vec = zeros(length(Ps))
for (i,P_) in enumerate(Ps)
  δ_base = 1 
  # set the Period.
  δ_ = δ_base* P_^(3/2)
  δ_vec[i] = δ_
  set_value(τ, P_)
  set_value(δ,δ_)

  optimize!(model)
  t_, p_, v_, f_, e_mech, e_damp, e_k, e_fr, e_met = packageresults()
  vpeak_vec[i] = maximum(v_)
  e_fr_vec[i]  = e_fr[end] 
  e_w_vec[i]  = maximum(e_mech)
  e_met_vec[i] = e_met[end]

  plot!(t_,p_,title="position",xlabel="time",subplot=1,legend=false)
  plot!(t_,v_,title="velocity",xlabel="time",subplot=2,legend=false)
  plot!(t_,f_,title="force",xlabel="time",subplot=3,legend=false)

  scatter!([fs[i]],[vpeak_vec[i]],subplot=4,ylimits=(0,1))

  if i ==1
    plot!(t_,e_mech,subplot=5, label="mech",linewidth=3)
    plot!(t_,e_damp,subplot=5, label = "damp")
    plot!(t_,e_met,subplot=5, label="met")
    plot!(t_,e_k,subplot=5,xlabel="time",ylabel="energy",label="kinetic",title="example energies")
    
  end
end
plot!(fs,vpeak_vec,subplot=4,ylimits=(0,1),xlimits=(0,maximum(fs)*1.5),color="black",xlabel="freq",ylabel="peak speed",legend=false,title="peak speed")

plot!(fs,e_fr_vec,subplot=6,label="fr",xlabel="freq")
plot!(fs,e_met_vec,subplot=6,color="green",xlabel="freq",xlimits = (0,maximum(fs)*1.5),label="e_mech_met", ylabel="energy fr,w",title="energy vs freq", legend=false)
f_loop

plot!(fs,δ_vec,ylabel="amplitude",xlabel="freq",subplot=7)

# dimensionalize by converting
fs = fs * 60 #bpm
δ_vec = δ_vec ./ δ_vec[1] .* 90 #deg
# for each fs, write text the frequency and the amplitude at each point in the plot
for (i,f_) in enumerate(fs)
  scatter!([f_],[δ_vec[i]], annotations = (f_,δ_vec[i], text("f=" * string(Int(round(f_))) *";deg=" * string(Int(round(δ_vec[i]))),:left, 5, "courier")), xlimits=(0,300),subplot=8, legend=false)
end

f_loop