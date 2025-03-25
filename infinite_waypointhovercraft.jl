using InfiniteOpt, Ipopt

xw = [1 4 6 1; 1 3 0 1] # positions
tw = [0, 25, 50, 60];    # times

m = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0));

@infinite_parameter(m, t in [0, 60], num_supports = 61)

@variables(m, begin
    # state variables
    x[1:2], Infinite(t)
    v[1:2], Infinite(t)
    # control variables
    u[1:2], Infinite(t), (start = 0)
end)

@objective(m, Min, ∫(u[1]^2 + u[2]^2, t))

@constraint(m, [i = 1:2], v[i](0) == 0)

@constraint(m, [i = 1:2], ∂(x[i], t) == v[i])
@constraint(m, [i = 1:2], ∂(v[i], t) == u[i])

@constraint(m, [i = 1:2, j = eachindex(tw)], x[i](tw[j]) == xw[i, j])

optimize!(m)
x_opt = value.(x);

using Plots
scatter(xw[1,:], xw[2,:], label = "Waypoints", background_color = :transparent)
plot!(x_opt[1], x_opt[2], label = "Trajectory")
xlabel!("x_1")
ylabel!("x_2")

using Test
@test termination_status(m) == MOI.LOCALLY_SOLVED
@test has_values(m)
@test x_opt isa Vector{<:Vector{<:Real}}