# Point mass, energy and time.

using InfiniteOpt, Ipopt, Plots, Distributions
include("plotpvu.jl")

## Understanding simple E + FR + T model

## Work and force-rate and time with
# point mass model
# function pointmassmodel(;c_fr = 0.05, c_t = 1,k_b = 5)
model = InfiniteModel(Ipopt.Optimizer)
c_t = 1
c_fr = 0.05
k_b = 1
@infinite_parameter(model, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))
@variable(model, p, Infinite(t), start = (t)->1*t)
@variable(model, v, Infinite(t))
@variable(model, fp >=0, Infinite(t))
@variable(model, fn <=0, Infinite(t))
@variable(model, fdot, Infinite(t))
@finite_parameter(model, δ == 1)

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
  e_mech = [[0];cumtrapz(t_, p_mech)]
  e_damp = [[0];cumtrapz(t_,k_b*v_.*v_)]
  e_k    = 1/2*1*v_.*v_
  return t_, p_, v_, f_, e_mech, e_damp, e_k
end

δs = [2]
f_loop = plot(layout = (2,2))
vpeaks = zeros(length(δs))
for (i,δ_) in enumerate(δs)
  set_value(δ, δ_)
  optimize!(model)
  t_, p_, v_, f_, e_mech, e_damp, e_k = packageresults()
  vpeaks[i] = maximum(v_)

  plot!(t_,p_,title="position",subplot=1)
  plot!(t_,v_,title="velocity",subplot=2)
  plot!(t_,f_,title="force",subplot=3)
  plot!(t_,[e_mech,e_damp,e_k,e_mech-e_damp-e_k], title="energy",legend = :topleft,subplot=4)
end

f_loop