# 2D Point Mass Walking Model with Two Force Actuators acting along the leg.
# if you want to run it without 
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# using Revise
using InfiniteOpt, Ipopt, Plots, Distributions, LinearAlgebra

function point_mass_walker()
    ## 2D Point Mass Walking Model
    model = InfiniteModel(Ipopt.Optimizer)

    # Model parameters
    c_fr = 0.05  # 0.05 creates linear forces. Force rate penalty coefficient
    c_t = 5.0    # Time penalty coefficient
    k_b = 0.0    # Damping coefficient
    g = 1     # Gravity
    ω0 = 0.3   # angular velocity 0
    # α = 0.35
    @finite_parameter(model,y_0==0.95)  # Initial leg length
    @finite_parameter(model,α==0.35)
    sl = 2*y_0*sin(α)

    # Infinite time parameter
    @infinite_parameter(model, τ ∈ [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))

    # State variables with bounds and initial guesses
    @variable(model, px, Infinite(τ), start = (t) -> value(sl)*t)  # x position
    @variable(model, py >= 0.1, Infinite(τ), start = (t) -> value(y_0) + cos(π*t)*.1)  # y position (must be positive)
    @variable(model, vx, Infinite(τ), start = (t) -> 1.0)  # x velocity
    @variable(model, vy, Infinite(τ), start = (t) -> 0.0)  # y velocity

    # Variable: Force magnitude along the leg, and must be positive (legs can only push)
    # Initial guesses for smooth bell-shaped force profiles
    @variable(model, F_trail >= 0, Infinite(τ), 
            start = (t) -> 1.0 * cos(2π*t) * (t < 0.5 ? 1.0 : 0.0))  # trailing leg force magnitude
    @variable(model, F_lead >= 0, Infinite(τ), 
            start = (t) -> 1.0 * -sin(2π*t) * (t <= 0.5 ? 0 : 1.0))  # leading leg force magnitude

    # Time scaling variable
    @variable(model, 0.001 <= t_f <= 10, start = 1)

    # Fixed leg contact positions; contact points are currently [y=0] beginning and end.
    @finite_parameter(model, P_trail_x==0.0)  # Trailing leg x-position
    @finite_parameter(model, P_trail_y==0.0)  # Trailing leg y-position
    @finite_parameter(model, P_lead_x==value(sl))  # Leading leg x-position
    @finite_parameter(model, P_lead_y==0.0)   # Leading leg y-position

    # Leg vectors (from contact points to COM)
    @expression(model, trail_leg_x, px - P_trail_x)
    @expression(model, trail_leg_y, py - P_trail_y)
    @expression(model, lead_leg_x, px - P_lead_x)
    @expression(model, lead_leg_y, py - P_lead_y)

    # Leg lengths
    @expression(model, trail_leg_length, sqrt(trail_leg_x^2 + trail_leg_y^2))
    @expression(model, lead_leg_length, sqrt(lead_leg_x^2 + lead_leg_y^2))

    # Unit vectors along each leg (with small epsilon to avoid division by zero)
    @expression(model, trail_unit_x, trail_leg_x / (trail_leg_length))
    @expression(model, trail_unit_y, trail_leg_y / (trail_leg_length))
    @expression(model, lead_unit_x, lead_leg_x / (lead_leg_length))
    @expression(model, lead_unit_y, lead_leg_y / (lead_leg_length))

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
            (px*vx + py*vy) / (trail_leg_length))

    # lead_leg_velocity
    @expression(model, lead_leg_velocity, 
                (-P_lead_x*vx + px*vx -P_lead_y*vy + py*vy) / (lead_leg_length))

    # Step 2: Compute mechanical power as force magnitude times leg-lengthening velocity
    # Note that positive power occurs when the leg is extending (trail_leg_velocity > 0)
    # Power slack variables for each leg
    @variable(model, pospower_trail >= 0, Infinite(τ))
    @variable(model, negpower_trail <= 0, Infinite(τ))
    @variable(model, pospower_lead >= 0, Infinite(τ))
    @variable(model, negpower_lead <= 0, Infinite(τ))

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

    # Step 4: System dynamics
    @constraint(model, ∂(px, τ) == t_f * vx)
    @constraint(model, ∂(py, τ) == t_f * vy)
    @constraint(model, ∂(vx, τ) == t_f * (Ftot_x))
    @constraint(model, ∂(vy, τ) == t_f * (Ftot_y - g))

    # Step 5: Force rate. Force dot variables
    ### begin
    @variable(model, Fdot_trail, Infinite(τ))
    @variable(model, Fdot_lead, Infinite(τ))

    # Force dot dot variables (split into positive and negative components)
    @variable(model, Fddot_trail_p >= 0, Infinite(τ))
    @variable(model, Fddot_trail_m >= 0, Infinite(τ))
    @variable(model, Fddot_lead_p >= 0, Infinite(τ))
    @variable(model, Fddot_lead_m >= 0, Infinite(τ))

    # Update force dynamics with scaling
    @variable(model, fdot_scale == 1)
    @variable(model, fddot_scale == 1)
    @constraint(model, ∂(F_trail, τ) == t_f * (Fdot_trail)/fdot_scale)
    @constraint(model, ∂(F_lead, τ) == t_f * (Fdot_lead)/fdot_scale)
    @constraint(model, ∂(Fdot_trail, τ) == t_f * ((Fddot_trail_p - Fddot_trail_m)/fddot_scale))
    @constraint(model, ∂(Fdot_lead, τ) == t_f * ((Fddot_lead_p - Fddot_lead_m)/fddot_scale))
    # have not needed the following two complimentarity constraints:
    # @constraint(model, Fddot_trail_p * Fddot_trail_m <= 1e-6) # complimentarity constraint
    # @constraint(model, Fddot_lead_p * Fddot_lead_m <= 1e-6)   # complimentarity constraint

    # Leg length constraints (forces only active when leg length <= 1)
    # Use complementarity constraints
    @constraint(model, F_trail * (trail_leg_length - 1) <= 0)
    @constraint(model, F_lead * (lead_leg_length - 1) <= 0)

    # Step 6: box constraints on initial/final states.
    # Initial and final boundary conditions for a complete step
    # Initial conditions
    @constraint(model, px(0) == 0)
    @constraint(model, py(0) == y_0)
    # @constraint(model, vx(0) == y_0 * ω0)
    @constraint(model, vy(0) == 0)  # Typically zero vertical velocity at start
    @constraint(model, vy(1) == 0)  # Same vertical velocity

    # Final conditions (symmetric gait)
    @constraint(model, px(1) == value(sl))  # One full step
    @constraint(model, py(1) == value(y_0))  # Same height at end
    @constraint(model, vx(1) == vx(0))  # Same horizontal velocity

    # Force boundary conditions
    @constraint(model, F_trail(1) == 0)
    @constraint(model, F_lead(0) == 0)
    @constraint(model, Fdot_lead(0) == Fdot_trail(1))
    @constraint(model, Fdot_trail(0) == Fdot_lead(1))

    @constraint(model,t_f == 1.2)

    # Objective function: minimize work, force rate, and time
    @expression(model, cost_work,integral(pospower_trail, τ) * t_f + integral(pospower_lead, τ) * t_f - 
            integral(negpower_trail, τ) * t_f - integral(negpower_lead, τ) * t_f)

    @expression(model, cost_fr, c_fr*integral(Fddot_trail_p, τ) * t_f +c_fr*integral(Fddot_lead_p,τ)*t_f + 
    c_fr*integral(Fddot_trail_m,τ) * t_f + c_fr*integral(Fddot_lead_m,τ)*t_f)

    @expression(model, cost_fr2, c_fr*integral(Fddot_trail_p.^2,τ) * t_f +c_fr*integral(Fddot_lead_p.^2,τ)*t_f + 
    c_fr*integral(Fddot_trail_m.^2,τ) * t_f + c_fr*integral(Fddot_lead_m.^2,τ)*t_f)

    @expression(model, cost_time, c_t*t_f)

    @objective(model, Min, cost_work+cost_fr)

    # Set solver parameters
    set_optimizer_attribute(model, "max_cpu_time", 120.0)
    set_optimizer_attributes(model, "tol" => 1e-3, "max_iter" => 500)
    # set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    # Solve the model
    optimize!(model)
    return model
end
# plot
function plot_results(model)
    f = plot(layout = (4,2), size = (800, 800))
    t_ = value(τ)*value(t_f)
    txt="work:"*string(round(value(cost_work),digits=2)) * " force rate:" * string(round(value(cost_fr),digits=2))
    plot!(value(px),value(py),subplot = 1,xlabel="x",ylabel="y",title=txt)
    plot!(t_,value(px),subplot = 2,ylabel="pxy",xlabel="time")
    plot!(t_,value(py),subplot = 2)
    plot!(t_,value(vx),subplot=3,label="vx",ylabel="vel")
    plot!(t_,value(vy),subplot=3,label="vy")
    plot!(t_,value(Ftrail_x), label="Ftrail_x",subplot=4,ylabel="force")
    plot!(t_,value(Ftrail_y), label="Ftrail_y",subplot=4)
    plot!(t_,value(Flead_y) + value(Ftrail_y), label="FY",subplot=4)
    plot!(t_,value(Flead_x), label="Flead_x",subplot=4)
    plot!(t_,value(Flead_y), label="Flead_y",subplot=4)

    plot!(t_,value(trail_leg_length),subplot=5,label="trail",linewidth=2,ylabel="leg length")
    plot!(t_,sqrt.((value(px) .- value(P_trail_x)).^2 + (value(py) .- value(P_trail_y)).^2),subplot=5,label="trailcomp")
    plot!(t_,value(lead_leg_length),subplot=5,label="lead")

    # plot fdotdot for each leg
    plot!(t_,value(Fddot_trail_p)-value(Fddot_trail_m),subplot=6,label="lead",ylabel="fraterate")
    plot!(t_,value(Fddot_lead_p)-value(Fddot_lead_m),subplot=6,label="lead")

    plot!(t_,value(trail_leg_velocity),subplot=7,label = "trail",ylabel="dlegdt")
    plot!(t_,value(lead_leg_velocity),subplot=7,label = "lead",ylabel="dlegdt")

    vx_dot = value(∂(vx,τ))
    vy_dot = value(∂(vy,τ))
    force_viol_x = vx_dot - value(Ftot_x)*value(t_f)
    force_viol_y = vy_dot - (value(Ftot_y) .- g)*value(t_f)
    plot!(t_,force_viol_x,subplot=8,label="dyn_viol_x")
    plot!(t_,force_viol_y,subplot=8,label="dyn_viol_y",xlabel="time")
    return f
end

f = plot_results()
f

# print out the constraints. 
for con in all_constraints(model)
             println("$(con): $(value(con))")
end