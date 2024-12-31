using InfiniteOpt, Ipopt, Plots, Distributions
include("plotpvu.jl")

## Hack together a simple ellipse drawing model
ellipse = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(ellipse, t in [0, 1], num_supports=10, derivative_method = OrthogonalCollocation(2))
#@variable(ellipse, T>=0, start = 1.)
@variable(ellipse, r, Infinite(t), start = 1)
@variable(ellipse, rdot, Infinite(t))
@variable(ellipse, fr, Infinite(t))
@variable(ellipse, frdot, Infinite(t))
@variable(ellipse, frddot, Infinite(t))
@variable(ellipse, theta, Infinite(t), start = t->pi*t)
@variable(ellipse, thetadot, Infinite(t))
@variable(ellipse, fn, Infinite(t))
@variable(ellipse, fndot, Infinite(t))
@variable(ellipse, fnddot, Infinite(t))
@variable(ellipse, x, Infinite(t))
@variable(ellipse, y, Infinite(t))

#relupower = log(1+exp(f*v))
#relupowern = log(1+exp(-f*v))
#relufddot = log(1+exp(fddot))
@objective(ellipse, Min, integral((fr*rdot)^2+(fn*r*thetadot)^2 + 
    1000*fnddot^2 + frddot^2,t)) # positive work & force rate squared
    
@constraint(ellipse, deriv(r,t) == rdot) # Dynamics
@constraint(ellipse, deriv(rdot,t)-r*thetadot^2 == -fr)  # F = ma for radial direction
@constraint(ellipse, deriv(fr,t) == frdot)
@constraint(ellipse, deriv(frdot,t) == frddot)
@constraint(ellipse, deriv(theta,t) == thetadot)
@constraint(ellipse, r*deriv(thetadot,t)+2*rdot*thetadot == fn) # normal direction
@constraint(ellipse, deriv(fn,t) == fndot)
@constraint(ellipse, deriv(fndot,t) == fnddot)

@constraint(ellipse, r(0) == r(1)) # Boundary conditions
@constraint(ellipse, theta(0) == 0)
@constraint(ellipse, theta(1) == pi) # periodicity

@constraint(ellipse, rdot(0) == rdot(1)) # Boundary conditions
@constraint(ellipse, fr(0) == fr(1)) # Boundary conditions
@constraint(ellipse, frdot(0) == frdot(1)) # Boundary conditions
@constraint(ellipse, frddot(0) == frddot(1))
@constraint(ellipse, theta(0) == 0)
@constraint(ellipse, theta(1) == pi) # periodicity
@constraint(ellipse, thetadot(0) == thetadot(1))
@constraint(ellipse, fn(0) == fn(1))
@constraint(ellipse, fndot(0) == fndot(1))
@constraint(ellipse, fnddot(0) == fnddot(1))

@constraint(ellipse, r(0) >= 1) # ellipse
@constraint(ellipse, r(1) >= 1)
@constraint(ellipse, r(0.5) <= 0.5)

@constraint(ellipse, r*cos(theta) == x)
@constraint(ellipse, r*sin(theta) == y)

optimize!(ellipse)
plot(value(x), value(y))
plot(supports(t), value(theta, ndarray=true))
plot(supports(t), value(r, ndarray=true))
plot(supports(t), value(fr, ndarray=true))
plot!(supports(t), value(fn, ndarray=true))

plot(supports(t), value(fndot))
plot(supports(t), value(fnddot))
plot(supports(t), value(frdot))
plot(supports(t), value(frddot))

plot(supports(t), value(fr).*value(rdot))
plot(supports(t), value(v,ndarray=true))
plot(supports(t), value(fp).*value(v))
plot!(supports(t), value(fn).*value(v))
plot!(supports(t), value(relufddot))
plot(supports(t), value(f).*value(v))
pfig = plotpvu(t,p,v,f)
plot!(pfig[3], supports(t), -value(fddotm, ndarray=true))
# This is a decent bell shape, although optimizer had a tough time
plot(supports(t), value(fn).*value(v))