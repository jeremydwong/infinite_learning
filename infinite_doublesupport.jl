# 2D Point mass optimization model with two force sources (legs)

using InfiniteOpt, Ipopt, Plots, Distributions

# Model setup
model = InfiniteModel(Ipopt.Optimizer)

# Cost weights
c_t = 1      # Cost of time
c_fr = 0.05  # Cost of force rate
k_b = 0      # Damping coefficient

# Position of the two "feet" (fixed points)
α = 0.35
sl = sin(2*α)
Ax, Ay = 0, 0  # Position of foot A
Bx, By = sl, 0   # Position of foot B

@infinite_parameter(model, t in [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))

# Position and velocity variables in 2D
@variable(model, px, Infinite(t), start = (t)-> Ax + (Bx-Ax)*t)
@variable(model, py, Infinite(t),start = 1)
@variable(model, vx, Infinite(t),start = (t)->sin(t))
@variable(model, vy, Infinite(t),start = (t)->sin(t))

# Forces from each leg in x and y directions
# Split into positive and negative components for slack vars.
@variable(model, vxp >= 0, Infinite(t))  # 
@variable(model, vxn <= 0, Infinite(t))  # 
@variable(model, vyp >= 0, Infinite(t))  # 
@variable(model, vyn <= 0, Infinite(t))  # 

@variable(model, fAxp >= 0, Infinite(t))  # Positive x-force from leg A
@variable(model, fAxn <= 0, Infinite(t))  # Negative x-force from leg A
@variable(model, fAyp >= 0, Infinite(t))  # Positive y-force from leg A
@variable(model, fAyn <= 0, Infinite(t))  # Negative y-force from leg A

@variable(model, fBxp >= 0, Infinite(t))  # Positive x-force from leg B
@variable(model, fBxn <= 0, Infinite(t))  # Negative x-force from leg B
@variable(model, fByp >= 0, Infinite(t))  # Positive y-force from leg B
@variable(model, fByn <= 0, Infinite(t))  # Negative y-force from leg B

# Total force components
@expression(model, fAx, fAxp + fAxn)
@expression(model, fAy, fAyp + fAyn)
@expression(model, fBx, fBxp + fBxn)
@expression(model, fBy, fByp + fByn)

# Force derivatives for smoothing
@variable(model, fAxdot, Infinite(t))
@variable(model, fAydot, Infinite(t))
@variable(model, fBxdot, Infinite(t))
@variable(model, fBydot, Infinite(t))

# Force second derivatives for smoothing cost
@variable(model, fAxddotp >= 0, Infinite(t))
@variable(model, fAxddotn <= 0, Infinite(t))
@variable(model, fAyddotp >= 0, Infinite(t))
@variable(model, fAyddotn <= 0, Infinite(t))

@variable(model, fBxddotp >= 0, Infinite(t))
@variable(model, fBxddotn <= 0, Infinite(t))
@variable(model, fByddotp >= 0, Infinite(t))
@variable(model, fByddotn <= 0, Infinite(t))

@variable(model, 0.001 <= t_f <= 10, start = 1)  # Time scaling

# Constraint: Forces must act along the leg directions
# For leg A: fAy/fAx = (py-Ay)/(px-Ax)
# This can be written as: fAy*(px-Ax) = fAx*(py-Ay)
@constraint(model, fAy * (px - Ax) == fAx * (py - Ay))

# For leg B: fBy/fBx = (py-By)/(px-Bx)
@constraint(model, fBy * (px - Bx) == fBx * (py - By))

# Dynamics
@constraint(model, ∂(px,t) == t_f * vx)
@constraint(model, ∂(py,t) == t_f * vy)
@constraint(model, ∂(vx,t) == t_f * ((fAx + fBx)))
@constraint(model, ∂(vy,t) == t_f * ((fAy + fBy)))
@constraint(model, ∂(fAx,t) == t_f * fAxdot)
@constraint(model, ∂(fAy,t) == t_f * fAydot)
@constraint(model, ∂(fBx,t) == t_f * fBxdot)
@constraint(model, ∂(fBy,t) == t_f * fBydot)
@constraint(model, ∂(fAxdot,t) == t_f * (fAxddotp + fAxddotn))
@constraint(model, ∂(fAydot,t) == t_f * (fAyddotp + fAyddotn))
@constraint(model, ∂(fBxdot,t) == t_f * (fBxddotp + fBxddotn))
@constraint(model, ∂(fBydot,t) == t_f * (fByddotp + fByddotn))

# Initial and final conditions
@finite_parameter(model, δx == sl)  # Final x position
@finite_parameter(model, δy == 0)  # Final y position
@finite_parameter(model, vx0 == 0.4)  # Initial x velocity
@finite_parameter(model, vy0 == -.1)    # Initial y velocity

@constraint(model, px(0) == Bx - Ax - .5*δx)
@constraint(model, py(0) == 1)
@constraint(model, px(1) == Bx-Ax+δx)
@constraint(model, py(1) == 1+δy)

@constraint(model, vx(0) == vx0)
@constraint(model, vy(0) == vy0)
@constraint(model, vx(1) == vx0)     # Final vx = initial vx
@constraint(model, vy(1) == -vy0)    # Final vy = -initial vy (bounce effect)

# Objective: mechanical work + force-rate costs
# @objective(model, Min, 
#     integral((fAxp - fAxn) * vx + (fAyp - fAyn) * vy + 
#              (fBxp - fBxn) * vx + (fByp - fByn) * vy + 
#              c_fr * ((fAxddotp - fAxddotn) + (fAyddotp - fAyddotn) +
#                      (fBxddotp - fBxddotn) + (fByddotp - fByddotn)), t) + c_t * t_f)

@objective(model, Min, 
    integral((fAxp) * vx + (fAyp) * vy,t) + integral((fBxp) * vx + (fByp) * vy,t) +
             c_fr * integral((fAxddotp),t) + integral(fAyddotp,t) + 
                     integral(fBxddotp,t) + integral(fByddotp, t))


set_optimizer_attribute(model, "max_cpu_time", 60.)
set_optimizer_attributes(model, "tol" => 1e-4, "max_iter" => 300)

optimize!(model)

f_ = plot(layout = (2,2))
# extract px and py
px_val = value(px)
py_val = value(py)
# now plot
plot!(px_val, py_val, xlabel="x", ylabel="y", title="2D Point Mass Model with Two Force Sources",subplot=1)
# Extract results and plot the forces in x and y
t_val = supports(t) * value(t_f)
fAx_val = value(fAxp) + value(fAxn)
fAy_val = value(fAyp) + value(fAyn)
fBx_val = value(fBxp) + value(fBxn)
fBy_val = value(fByp) + value(fByn)

# now plot in the first subplot

plot!(t_val, fAx_val, label="fAx",subplot = 2)
plot!(t_val, fAy_val, label="fAy",subplot = 2)
plot!(t_val, fBx_val, label="fBx",subplot = 2)
plot!(t_val, fBy_val, label="fBy", xlabel="Time", ylabel="Force", title="Forces in x and y directions",subplot = 2)
f_
