# Point mass, energy and time.

using InfiniteOpt, Ipopt, Plots, Distributions

function cumtrapz(t, y) # used for plotting.
  return cumsum((y[1:end-1] + y[2:end]) .* diff(t))/2
end

## Understanding simple E + FR + T model

## Work and force-rate and time with
# point mass model
# function pointmassmodel(;c_fr = 0.05, c_t = 1,k_b = 5)
model = InfiniteModel(Ipopt.Optimizer)
# c_t = 1
c_fr = 0.05
k_b = 0
@infinite_parameter(model, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))
@variable(model, p, Infinite(t), start = (t)->1*t)
@variable(model, v, Infinite(t))
@variable(model, f, Infinite(t))
@variable(model, pospower, Infinite(t))
@variable(model, negpower, Infinite(t))
@expression(model, mechpower, (f) * v)

@constraint(model, pospower>=mechpower)
@constraint(model, pospower>=0)
@constraint(model, negpower<=mechpower)
@constraint(model, negpower<=0)

@variable(model, fdot, Infinite(t))

@finite_parameter(model, δ == .1)
@finite_parameter(model, c_t == 10)

@variable(model, fddotp >= 0, Infinite(t))
@variable(model, fddotm <= 0, Infinite(t))

@variable(model, 0.001 <= t_f <= 10, start = 1) # scaling of time by tf.

@constraint(model, ∂(p,t)        == t_f * v) # Dynamics
@constraint(model, ∂(v,t)        == t_f * ((f) - k_b*v))
@constraint(model, (∂(f,t))  == t_f * fdot) 
@constraint(model, ∂(fdot, t)    == t_f * (fddotp + fddotm))

set_value(c_t, 5) #sensitive! 1-4.
# objective: positive power - negative power + force rate + time
@objective(model, Min, integral(pospower, t)*t_f - integral(negpower, t)*t_f + integral(c_fr*(fddotp - fddotm),t)*t_f + c_t * t_f) # abs work and force rate 

# Boundary conditions
@constraint(model, p(0) == 0) 
@constraint(model, p(1) == δ)
@constraint(model, v(0) == 0)
@constraint(model, v(1) == 0)
@constraint(model, f(0) == 0)
@constraint(model, f(1) == 0)
@constraint(model, fdot(0) == 0)
@constraint(model, fdot(1) == 0)

# optimizer attributes
set_optimizer_attribute(model, "max_cpu_time", 60.)
set_optimizer_attributes(model, "tol" => 1e-4, "max_iter" => 150)
set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")

# optimize the model! Find decision variables that minimize the objective function. 
optimize!(model)

# allow warm-starting
set_optimizer_attribute(model, "warm_start_init_point", "yes")

# define function to package results
function packageresults() # used for plotting to gather all variables.
  """
  Packageresults gathers all the variables from the model and returns them in a format suitable for plotting.
  It returns time, position, velocity, force, mechanical energy, damping energy, and kinetic
  energy.
  
  """
  v_ = value(v)
  p_ = value(p)
  t_ = supports(t)*value(t_f)
  f_ = value(f)
  p_mech = (f_.*v_)
  e_mech = [[0];cumtrapz(t_, p_mech)]
  e_damp = [[0];cumtrapz(t_,k_b*v_.*v_)]
  e_k    = 1/2*1*v_.*v_
  return t_, p_, v_, f_, e_mech, e_damp, e_k
end

# here we reoptimize for different values of δ, which is the distance to travel.
δs = [1]
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
  plot!(t_,[e_mech,e_damp,e_k,e_mech-e_damp-e_k], labels=["mech" "damp" "kin" "err"], title="energy",legend = :topleft,subplot=4)
  err = e_mech-e_damp-e_k
  print("error in energy: ", round(err[end],digits=2), "J\n")
  print("% error of total mechanical: ", round(err[end]/maximum(e_mech)*100,digits=2), "%\n")
end
f_loop