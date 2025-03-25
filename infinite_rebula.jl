# 2D Point Mass Walking Model with Two Force Actuators
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using Revise
using InfiniteOpt, Ipopt, Plots, Distributions, LinearAlgebra

## 2D Point Mass Walking Model
model = InfiniteModel(Ipopt.Optimizer)

# Model scalar parameters. 
c_fr = 0.05  # Force rate penalty coefficient
c_t = 5.0    # Time penalty coefficient
k_b = 0.0    # Damping coefficient
g   = 1     # Gravity

α   = 0.35
y_0 = 0.95  # Initial leg length
sl = 2*y_0*sin(α)
# optional initial angular velocity. 
ω0  = 0.3   # angular velocity 0

# Infinite time parameters
@infinite_parameter(model, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))

# State variables with bounds and initial guesses
@variable(model, px, Infinite(t), start = (t) -> value(sl)*t)  # x position
@variable(model, py >= 0.1, Infinite(t), start = (t) -> y_0 + cos(π*t)*.1)  # y position (must be positive)
@variable(model, vx, Infinite(t), start = (t) -> 1.0)  # x velocity
@variable(model, vy, Infinite(t), start = (t) -> 0.0)  # y velocity

# Variable: Force magnitude along the leg, and must be positive (legs can only push)
# Initial guesses for smooth bell-shaped force profiles
# @variable(model, F_trail >= 0, Infinite(t), 
#           start = (t) -> 1.0 * cos(π*t)^2 * (t < 0.4 ? 1.0 : 0.0))  # trailing leg force magnitude
# @variable(model, F_lead >= 0, Infinite(t), 
#           start = (t) -> 1.0 * sin(π*t)^2 * (t >= 0.4 ? 1.0 : 0.0))  # leading leg force magnitude
@variable(model, F_trail >= 0, Infinite(t), 
          start = (t) -> 1.0 * cos(2π*t) * (t < 0.5 ? 1.0 : 0.0))  # trailing leg force magnitude
@variable(model, F_lead >= 0, Infinite(t), 
          start = (t) -> 1.0 * -sin(2π*t) * (t <= 0.5 ? 0 : 1.0))  # leading leg force magnitude


# Force dot variables
@variable(model, Fdot_trail, Infinite(t))
@variable(model, Fdot_lead, Infinite(t))

# Force dot dot variables (split into positive and negative components)
@variable(model, Fddot_trail_p >= 0, Infinite(t))
@variable(model, Fddot_trail_m >= 0, Infinite(t))
@variable(model, Fddot_lead_p >= 0, Infinite(t))
@variable(model, Fddot_lead_m >= 0, Infinite(t))

# Time scaling variable
@variable(model, 0.001 <= t_f <= 10, start = 1)

# Fixed leg contact positions
# For simplicity, contact points are at y=0
@expression(model, P_trail_x, 0.0)  # Trailing leg ground contact x-position
@expression(model, P_trail_y, 0.0)  # Trailing leg ground contact y-position
@expression(model, P_lead_x, sl)  # Leading leg ground contact x-position
@expression(model, P_lead_y, 0.0)   # Leading leg ground contact y-position

# Leg vectors (from contact points to COM)
@expression(model, trail_leg_x, px - P_trail_x)
@expression(model, trail_leg_y, py - P_trail_y)
@expression(model, lead_leg_x, px - P_lead_x)
@expression(model, lead_leg_y, py - P_lead_y)

# Leg lengths
@expression(model, trail_leg_length, sqrt(trail_leg_x^2 + trail_leg_y^2))
@expression(model, lead_leg_length, sqrt(lead_leg_x^2 + lead_leg_y^2))

# Unit vectors along each leg (with small epsilon to avoid division by zero)
epsilon = 1e-6
@expression(model, trail_unit_x, trail_leg_x / (trail_leg_length + epsilon))
@expression(model, trail_unit_y, trail_leg_y / (trail_leg_length + epsilon))
@expression(model, lead_unit_x, lead_leg_x / (lead_leg_length + epsilon))
@expression(model, lead_unit_y, lead_leg_y / (lead_leg_length + epsilon))

# Force components for each leg
@expression(model, Ftrail_x, F_trail * trail_unit_x)
@expression(model, Ftrail_y, F_trail * trail_unit_y)
@expression(model, Flead_x,  F_lead * lead_unit_x)
@expression(model, Flead_y,  F_lead * lead_unit_y)

# Total force components
@expression(model, Ftot_x, Ftrail_x + Flead_x)
@expression(model, Ftot_y, Ftrail_y + Flead_y)  # Excluding gravity

# trail_leg_velocity
@expression(model, trail_leg_velocity, 
           (px*vx + py*vy) / (trail_leg_length + epsilon))

# lead_leg_velocity
@expression(model, lead_leg_velocity, 
            (-P_lead_x*vx + px*vx -P_lead_y*vy + py*vy) / (lead_leg_length + epsilon))

# Step 2: Compute mechanical power as force magnitude times leg-lengthening velocity
# Note that positive power occurs when the leg is extending (trail_leg_velocity > 0)
# Power slack variables for each leg
@variable(model, pospower_trail >= 0, Infinite(t))
@variable(model, negpower_trail <= 0, Infinite(t))
@variable(model, pospower_lead >= 0, Infinite(t))
@variable(model, negpower_lead <= 0, Infinite(t))

@expression(model, mechpower_trail, F_trail * trail_leg_velocity)
@expression(model, mechpower_lead, F_lead * lead_leg_velocity)

# Step 3: Update the power splitting constraints
# Power is positive when doing work (leg extending)
@constraint(model, pospower_trail >= mechpower_trail)
@constraint(model, pospower_trail >= 0)
@constraint(model, negpower_trail <= mechpower_trail)
@constraint(model, negpower_trail <= 0)

@constraint(model, pospower_lead >= mechpower_lead)
@constraint(model, pospower_lead >= 0)
@constraint(model, negpower_lead <= mechpower_lead)
@constraint(model, negpower_lead <= 0)

# System dynamics
@constraint(model, ∂(px, t) == t_f * vx)
@constraint(model, ∂(py, t) == t_f * vy)
@constraint(model, ∂(vx, t) == t_f * (Ftot_x))
@constraint(model, ∂(vy, t) == t_f * (Ftot_y - g))

# Update force dynamics with scaling
@variable(model, fdot_scale == 1)
@variable(model, fddot_scale == 1)
@constraint(model, ∂(F_trail, t) == t_f * (Fdot_trail)/fdot_scale)
@constraint(model, ∂(F_lead, t) == t_f * (Fdot_lead)/fdot_scale)
@constraint(model, ∂(Fdot_trail, t) == t_f * ((Fddot_trail_p - Fddot_trail_m)/fddot_scale))
@constraint(model, ∂(Fdot_lead, t) == t_f * ((Fddot_lead_p - Fddot_lead_m)/fddot_scale))
# @constraint(model, Fddot_trail_p * Fddot_trail_m <= 1e-6) # complimentarity constraint
# @constraint(model, Fddot_lead_p * Fddot_lead_m <= 1e-6)   # complimentarity constraint

# Leg length constraints (forces only active when leg length <= 1)
# Use complementarity constraints
@constraint(model, [t=supports(t)], F_trail * (trail_leg_length - 1) <= 0)
@constraint(model, [t=supports(t)], F_lead * (lead_leg_length - 1) <= 0)

# Ensure vertical force components are upward (or zero)
# These constraints help ensure the legs are supporting the mass against gravity
@constraint(model, [t=supports(t)], Ftrail_y >= 0)
@constraint(model, [t=supports(t)], Flead_y >= 0)

# Initial and final boundary conditions for a complete step
# Initial conditions
@constraint(model, px(0) == 0)
@constraint(model, py(0) == y_0)
@constraint(model, vx(0) == y_0 * ω0)
@constraint(model, vy(0) == 0)  # Typically zero vertical velocity at start
@constraint(model, vy(1) == 0)  # Same vertical velocity

# Final conditions (symmetric gait)
@constraint(model, px(1) == sl)  # One full step
@constraint(model, py(1) == y_0)  # Same height at end
@constraint(model, vx(1) == vx(0))  # Same horizontal velocity

# Force boundary conditions
@constraint(model, F_trail(1) == 0)
@constraint(model, F_lead(0) == 0)
@constraint(model, Fdot_lead(0) == Fdot_trail(1))
@constraint(model, Fdot_trail(0) == Fdot_lead(1))

@constraint(model,t_f == 1.0)

# Objective function: minimize work, force rate, and time
@expression(model, cost_work,integral(pospower_trail, t) * t_f + integral(pospower_lead, t) * t_f - 
          integral(negpower_trail, t) * t_f - integral(negpower_lead, t) * t_f)

@expression(model, cost_fr, c_fr*integral(Fddot_trail_p,t) * t_f +c_fr*integral(Fddot_lead_p,t)*t_f + 
c_fr*integral(Fddot_trail_m,t) * t_f + c_fr*integral(Fddot_lead_m,t)*t_f)

@objective(model, Min, cost_work+cost_fr)

# Set solver parameters
set_optimizer_attribute(model, "max_cpu_time", 120.0)
set_optimizer_attributes(model, "tol" => 1e-3, "max_iter" => 200)
# set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
set_optimizer_attribute(model, "warm_start_init_point", "yes")

# Solve the model
optimize!(model)

# plot
f = plot(layout = (4,2), size = (800, 800))
t_ = value(t)*value(t_f)
txt="work:"*string(round(value(cost_work),digits=2)) * " force rate:" * string(round(value(cost_fr),digits=2))
plot!(value(px),value(py),subplot = 1,xlabel="x",ylabel="y",title=txt)
plot!(t_,value(px),subplot = 2,ylabel="pxy",xlabel="time")
plot!(t_,value(py),subplot = 2)
plot!(t_,value(vx),subplot=3,label="vx",ylabel="vel")
plot!(t_,value(vy),subplot=3,label="vy")
plot!(t_,value(Ftrail_x), label="Ftrail_x",subplot=4,ylabel="force")
plot!(t_,value(Ftrail_y), label="Ftrail_y",subplot=4)
plot!(t_,value(Flead_x), label="Flead_x",subplot=4)
plot!(t_,value(Flead_y), label="Flead_y",subplot=4)

plot!(t_,value(trail_leg_length),subplot=5,label="trail",linewidth=2,ylabel="leg length")
plot!(t_,sqrt.((value(px) .- P_trail_x).^2 + (value(py) .- P_trail_y).^2),subplot=5,label="trailcomp")
plot!(t_,value(lead_leg_length),subplot=5,label="lead")

# plot fdotdot for each leg
plot!(t_,value(Fddot_trail_p)+value(Fddot_trail_m),subplot=6,label="lead",ylabel="fraterate")
plot!(t_,value(Fddot_lead_p)+value(Fddot_lead_m),subplot=6,label="lead")

plot!(t_,value(trail_leg_velocity),subplot=7,label = "trail",ylabel="dlegdt")
plot!(t_,value(lead_leg_velocity),subplot=7,label = "lead",ylabel="dlegdt")

vx_dot = value(∂(vx,t)*t_f)
vy_dot = value(∂(vy,t)*t_f)
force_viol_x = vx_dot - value(t_f) * value(Ftot_x)
force_viol_y = vy_dot - value(t_f) * value(Ftot_y) .+ value(t_f) * g
plot!(t_,force_viol_x,subplot=8,label="dyn_viol_x")
plot!(t_,force_viol_y,subplot=8,label="dyn_viol_y",xlabel="time")

f

# function debug_solution()
#   t_ = supports(t)
  
#   # Check complementarity constraints
#   trail_length_viol = value(trail_leg_length) .- 1.0
#   lead_length_viol = value(lead_leg_length) .- 1.0
  
#   # Check force magnitudes
#   trail_forces = value(F_trail)
#   lead_forces = value(F_lead)
  
#   # Check product terms (should be close to zero for complementarity)
#   trail_comp = trail_forces .* trail_length_viol
#   lead_comp = lead_forces .* lead_length_viol
  
#   println("Max trail length violation: ", maximum(trail_length_viol))
#   println("Max lead length violation: ", maximum(lead_length_viol))
#   println("Max trail force: ", maximum(trail_forces))
#   println("Max lead force: ", maximum(lead_forces))
#   println("Max trail complementarity violation: ", maximum(abs.(trail_comp)))
#   println("Max lead complementarity violation: ", maximum(abs.(lead_comp)))
  
#   # Check dynamics violations
#   px_dot = value(∂(px,t)*t_f)
#   py_dot = value(∂(py,t)*t_f)
#   vx_dot = value(∂(vx,t)*t_f)
#   vy_dot = value(∂(vy,t)*t_f)
  
#   dyn_viol_vx = px_dot - value(t_f) * value(vx)
#   dyn_viol_vy = py_dot - value(t_f) * value(vy)
#   force_viol_x = vx_dot - value(t_f) * value(Ftot_x)
#   force_viol_y = vy_dot - value(t_f) * value(Ftot_y) .- value(t_f) * g
  
#   vleg_trail = value(trail_leg_velocity)
#   vleg_lead = value(lead_leg_velocity)

#   println("Max dynamics violation in x-position: ", maximum(abs.(dyn_viol_x)))
#   println("Max dynamics violation in y-position: ", maximum(abs.(dyn_viol_y)))
#   println("Max force violation in x-direction: ", maximum(abs.(force_viol_x)))
#   println("Max force violation in y-direction: ", maximum(abs.(force_viol_y)))

#   plot(force_viol_x, label="Force violation in x-direction", xlabel="Supports")
#   plot!(force_viol_y, label="Force violation in y-direction")
# end

# # debug_solution()
