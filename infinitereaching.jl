# note: this is originally modelling from Art Kuo, 
# I may have inadvertently edited, but was using for reference.
# Infinite reaching

using InfiniteOpt, Ipopt, Plots, Distributions
include("plotpvu.jl")
# Minimum jerk/variance/fdot point mass model
# Arbitrary third-order model (with damping) is sufficient to 
# produce bell-shaped trajectories with fdot squared
# extra damping affects the actuation but not the motion

hand = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(hand, t in [0, 1], num_supports=40, derivative_method = OrthogonalCollocation(2))
@finite_parameter(hand, B == 0)
@variable(hand, p, Infinite(t))
@variable(hand, v, Infinite(t))
@variable(hand, F, Infinite(t))
@variable(hand, Fdot, Infinite(t))

@objective(hand, Min, integral(Fdot.^2,t))

@constraint(hand, deriv(p,t) == v)   # triple integrator with damping
@constraint(hand, deriv(v,t) == -B*v + F)
@constraint(hand, deriv(F,t) == Fdot)

@constraint(hand, p(0) == 0)
@constraint(hand, p(1) == 1)
@constraint(hand, v(0) == 0)
@constraint(hand, v(1) == 0)
@constraint(hand, F(0) == 0)
@constraint(hand, F(1) == 0)

optimize!(hand)

pfig = plot(layout=(1,2))
plot!(pfig[1],supports(t), value(v))
plot!(pfig[2], supports(t), value(F))

pfig = plot()
for Bval in 0:0.2:2
    set_value( B, Bval)
    optimize!(hand)
    plotpvu!(pfig, t, (p, v,F)..., title="B = $Bval")
end
display(pfig)

## Third order eye model Harris & Wolpert
# three time constants, 224, 13, 10 ms
# (Need 3rd order to get bell shape)
using JuMP, Ipopt, ControlSystems

# Eye movement parameters
th = 0.05 # hold time in continuous time
mytf = 0.1  # final time of simuilation

# Start with transfer function, convert to continuous-time state space, then discrete time
tfeye = tf(1, [0.224, 1])*tf(1, [0.013, 1])*tf(1, [0.010, 1]) # Harris & Wolpert 3rd-order eye model
eyeden = tfeye.matrix[1].den  # denominator coefficients in ascending order: b[0] + b[1]*s + b[2]*s^2

eyesdn = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyesdn, t in [0, mytf], num_supports=40, derivative_method = OrthogonalCollocation(2))
@infinite_parameter(eyesdn, w ~ Normal(0, 0.1^2), num_supports=20)
@variable(eyesdn, p, Infinite(t,w), start = (t,w)-> t <= th ? 1*t/th : 1.)
@variable(eyesdn, v, Infinite(t,w))
@variable(eyesdn, a, Infinite(t,w))
@variable(eyesdn, u, Infinite(t))
@objective(eyesdn, Min, integral(expect(p^2,w), t,th,mytf)) # expected variance 

@constraint(eyesdn, deriv(p,t) == v) # Dynamics
@constraint(eyesdn, deriv(v,t) == a)
@constraint(eyesdn, deriv(a,t) == -eyeden[0]/eyeden[3]*p -eyeden[1]/eyeden[3]*v - eyeden[2]/eyeden[3]*a + 1/eyeden[3]*u*(1+8*w))

@constraint(eyesdn, p(0,w) == 0) # Boundary conditions
@constraint(eyesdn, expect(p(th,w),w) == 1)
@constraint(eyesdn, v(0,w) == 0)
@constraint(eyesdn, expect(v(th,w),w) == 0) # need to bring velocity to zero
@constraint(eyesdn, a(0,w) == 0) # this helps make start bell-shaped
@constraint(eyesdn, expect(a(th,w),w) == 0) # helps to bring acceleration to zero
#@constraint(eyesdn, hold, expect(v,w) == 0, DomainRestrictions(t => [th,mytf])) # don't need a hold constraint

optimize!(eyesdn)
traces = (p,v,u)
pfig=plotpvu(t, traces..., title="min var")
@objective(eyesdn, Min, 0) # also works with no objective whatsoever
plotpvu!(pfig,eyesdn, t,p,v,u, "min 0")


## SDN with deterministic system
# zero-mean noise doesn't affect the expected position of a linear model
# so you can actually just do optimal control on the mean-square position
# (and weight that integral by the noise if you want to, but unnecessary)
eyesdn = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyesdn, t in [0, mytf], num_supports=30, derivative_method = OrthogonalCollocation(2))
@variable(eyesdn, p, Infinite(t))
@variable(eyesdn, v, Infinite(t))
@variable(eyesdn, a, Infinite(t))
@variable(eyesdn, u, Infinite(t))
@objective(eyesdn, Min, integral((p-1)^2, t,th,mytf)) # expected variance 

@constraint(eyesdn, deriv(p,t) == v)
@constraint(eyesdn, deriv(v,t) == a)
@constraint(eyesdn, deriv(a,t) == -eyeden[0]/eyeden[3]*p -eyeden[1]/eyeden[3]*v - eyeden[2]/eyeden[3]*a + 1/eyeden[3]*u)

@constraint(eyesdn, p(0) == 0)
@constraint(eyesdn, p(th) == 1)
@constraint(eyesdn, v(0) == 0)
@constraint(eyesdn, v(th) == 0)
@constraint(eyesdn, a(0) == 0)
@constraint(eyesdn, a(th) == 0)
#@constraint(eyesdn, hold, v == 0, DomainRestrictions(t => [th,mytf]))

optimize!(eyesdn)
pfig = plotpvu(t, p, v, u, title="min var")
@objective(eyesdn, Min, 0) # don't even optimize, just hold and get similar results
plotpvu!(pfig, t, p, v, u)


## Eye with fdoubledot rectified value
# You can also do away with the specific time constants and just do a pure integrator, and
# still get good results (no hold period), smoother bell shape
eyefdd = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyefdd, t in [0, mytf], num_supports=40, derivative_method = OrthogonalCollocation(2))
@variable(eyefdd, p, Infinite(t))
@variable(eyefdd, v, Infinite(t))
@variable(eyefdd, f, Infinite(t))
@variable(eyefdd, fdot, Infinite(t))
@variable(eyefdd, fddotp >= 0, Infinite(t))
@variable(eyefdd, fddotm >= 0, Infinite(t))
@objective(eyefdd, Min, integral((p-1)^2+fddotp, t)) # mse + fddot 

@constraint(eyefdd, deriv(p,t) == v) # Dynamics
@constraint(eyefdd, deriv(v,t) == f)
@constraint(eyefdd, deriv(f,t) == fdot)
@constraint(eyefdd, deriv(fdot,t) == fddotp - fddotm)

@constraint(eyefdd, p(0) == 0) # Boundary conditions
@constraint(eyefdd, p(mytf) == 1)
@constraint(eyefdd, v(0) == 0)
@constraint(eyefdd, v(mytf) == 0)
@constraint(eyefdd, f(0) == 0)
@constraint(eyefdd, f(mytf) == 0)
@constraint(eyefdd, fdot(0) == 0)
@constraint(eyefdd, fdot(mytf) == 0)

optimize!(eyefdd)
pfig = plotpvu(t,p,v,fddotp)
plot!(pfig[3], supports(t), -value(fddotm, ndarray=true))


## Eye with fdoubledot squared
# using pure integrator, pretty much works no matter the weighting
eyefdd = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyefdd, t in [0, mytf], num_supports=40, derivative_method = OrthogonalCollocation(2))
@finite_parameter(eyefdd, weighting == 0)
@variable(eyefdd, p, Infinite(t))
@variable(eyefdd, v, Infinite(t))
@variable(eyefdd, f, Infinite(t))
@variable(eyefdd, fdot, Infinite(t))
@variable(eyefdd, fddot, Infinite(t))
#@objective(eyefdd, Min, integral((p-1)^2,t))
#@objective(eyefdd, Min, integral(fddot^2,t))
@objective(eyefdd, Min, integral((1-weighting)*(p-1)^2+weighting*fddot^2, t)) # mse + fddot 
#@objective(eyefdd, Min, integral(abs(fddot), t)) # mse + fddot 

@constraint(eyefdd, deriv(p,t) == v) # Dynamics
@constraint(eyefdd, deriv(v,t) == f)
@constraint(eyefdd, deriv(f,t) == fdot)
@constraint(eyefdd, deriv(fdot,t) == fddot)

@constraint(eyefdd, p(0) == 0) # Boundary conditions
@constraint(eyefdd, p(mytf) == 1)
@constraint(eyefdd, v(0) == 0)
@constraint(eyefdd, v(mytf) == 0)
@constraint(eyefdd, f(0) == 0)
@constraint(eyefdd, f(mytf) == 0)
@constraint(eyefdd, fdot(0) == 0) # this is kind of optional
@constraint(eyefdd, fdot(mytf) == 0)

optimize!(eyefdd)
pfig = plotpvu(t, p, v, fddot)

weightings = 0:0.1:1 # make the plot for position squared vs fddot^2
pfig = plot(layout=@layout([° °; ° _]))
for i in eachindex(weightings)
    set_value(weighting, weightings[i])
#    @objective(eyesdn, Min, integral((1-weighting)*(p-1)^2+weighting*fddot^2, t,th,mytf))
    optimize!(eyefdd)
    plotpvu!(pfig, t,p,v,fddot)
end
display(pfig)


## this actually worked pretty well, surprisingly
# minimum v^2 during hold period; no end velocity or acceleration, no hold constraint
# a bit wiggly at the end, but there's no hold!
eyesdn = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyesdn, t in [0, mytf], num_supports=40, derivative_method = OrthogonalCollocation(2))
@infinite_parameter(eyesdn, w ~ Normal(0, 0.1^2), num_supports=20)
@variable(eyesdn, p, Infinite(t,w))
@variable(eyesdn, v, Infinite(t,w))
@variable(eyesdn, a, Infinite(t,w))
@variable(eyesdn, u, Infinite(t))
@objective(eyesdn, Min, integral(expect(v^2,w), t,th,mytf)) # expected variance 

@constraint(eyesdn, deriv(p,t) == v) # Dynamics
@constraint(eyesdn, deriv(v,t) == a)
@constraint(eyesdn, deriv(a,t) == -eyeden[0]/eyeden[3]*p -eyeden[1]/eyeden[3]*v - eyeden[2]/eyeden[3]*a + 1/eyeden[3]*u*(1+8*w))

@constraint(eyesdn, p(0,w) == 0) # Boundary conditions
@constraint(eyesdn, expect(p(th,w),w) == 1)
@constraint(eyesdn, v(0,w) == 0)
#@constraint(eyesdn, expect(v(th,w),w) == 0) # don't need it
@constraint(eyesdn, a(0,w) == 0)
#@constraint(eyesdn, expect(a(th,w),w) == 0) # don't need it
#@constraint(eyesdn, hold, expect(v,w) == 0, DomainRestrictions(t => [th,mytf])) # don't need it

optimize!(eyesdn)
pfig=plotpvu(t,p,v,u, title="min vvar")

## Force-rate and cost of time, using absolute value of force-rate which doesn't seem to matter a lot
eyefdd = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyefdd, t in [0, 1], num_supports=40, derivative_method = OrthogonalCollocation(2))
@variable(eyefdd, T>=0, start = 1.)
@variable(eyefdd, p, Infinite(t), start = (t)->1*t)
@variable(eyefdd, v, Infinite(t))
@variable(eyefdd, f, Infinite(t))
@variable(eyefdd, fdot, Infinite(t))
@variable(eyefdd, fddotp >= 0, Infinite(t))
@variable(eyefdd, fddotm >= 0, Infinite(t))
@objective(eyefdd, Min, integral((p-1)^2+1*fddotp, t)+10000.5*T) # mse + fddot 

@constraint(eyefdd, deriv(p,t) == T*v) # Dynamics
@constraint(eyefdd, deriv(v,t) == T*f)
@constraint(eyefdd, deriv(f,t) == T*fdot)
@constraint(eyefdd, deriv(fdot,t) == T*(fddotp - fddotm))

@constraint(eyefdd, p(0) == 0) # Boundary conditions
@constraint(eyefdd, p(1) == 1)
@constraint(eyefdd, v(0) == 0)
@constraint(eyefdd, v(1) == 0)
@constraint(eyefdd, f(0) == 0)
@constraint(eyefdd, f(1) == 0)
@constraint(eyefdd, fdot(0) == 0)
@constraint(eyefdd, fdot(1) == 0)

optimize!(eyefdd)
plot(supports(t)*value(T), value(v,ndarray=true))
pfig = plotpvu(t,p,v,fddotp)
plot!(pfig[3], supports(t), -value(fddotm, ndarray=true))

## Work and force-rate and time with
# point mass model
eyewft = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyewft, t in [0, 1], num_supports=40, derivative_method = OrthogonalCollocation(2))
#@variable(eyewft, T>=0, start = 1.)
@variable(eyewft, p, Infinite(t), start = (t)->1*t)
@variable(eyewft, v, Infinite(t))
@variable(eyewft, fp >=0, Infinite(t))
@variable(eyewft, fn <=0, Infinite(t))
@variable(eyewft, fdot, Infinite(t))
#@variable(eyewft, pow 0, Infinite(t))
#@variable(eyewft, pown <= 0, Infinite(t))
@variable(eyewft, fddotp >= 0, Infinite(t))
@variable(eyewft, fddotm >= 0, Infinite(t))
@objective(eyewft, Min, integral(fp*v - fn*v + fddotp,t)) # positive work 

@constraint(eyewft, deriv(p,t) == v) # Dynamics
@constraint(eyewft, deriv(v,t) == fp + fn)
@constraint(eyewft, deriv(fp,t)-deriv(fn,t) == fdot)
@constraint(eyewft, deriv(fdot,t) == (fddotp - fddotm))
#@constraint(eyewft, f*v == powp + pown)

@constraint(eyewft, p(0) == 0) # Boundary conditions
@constraint(eyewft, p(1) == 1)
@constraint(eyewft, v(0) == 0)
@constraint(eyewft, v(1) == 0)
@constraint(eyewft, fp(0)+fn(0) == 0)
@constraint(eyewft, fp(1)+fn(1) == 0)
@constraint(eyewft, fdot(0) == 0)
@constraint(eyewft, fdot(1) == 0)

optimize!(eyewft)
plot(supports(t), value(v,ndarray=true))
plot(supports(t), value(fp).*value(v))
plot!(supports(t), value(fn).*value(v))
pfig = plotpvu(t,p,v,f)
plot!(pfig[3], supports(t), -value(fddotm, ndarray=true))
plot(supports(t), value(fn).*value(v))

## Point mass with soft relu power
## Work and force-rate and time with
# point mass model
eyewft = InfiniteModel(Ipopt.Optimizer)
@infinite_parameter(eyewft, t in [0, 1], num_supports=40, derivative_method = OrthogonalCollocation(2))
#@variable(eyewft, T>=0, start = 1.)
@variable(eyewft, p, Infinite(t), start = (t)->1*t)
@variable(eyewft, v, Infinite(t))
#@variable(eyewft, fp >=0, Infinite(t))
#@variable(eyewft, fn <=0, Infinite(t))
@variable(eyewft, f, Infinite(t))
@variable(eyewft, fdot, Infinite(t))
#@variable(eyewft, pow 0, Infinite(t))
#@variable(eyewft, pown <= 0, Infinite(t))
@variable(eyewft, fddot, Infinite(t))
#@variable(eyewft, fddotp >= 0, Infinite(t))
#@variable(eyewft, fddotm >= 0, Infinite(t))
relupower = log(1+exp(f*v))
relupowern = log(1+exp(-f*v))
relufddot = log(1+exp(fddot))
@objective(eyewft, Min, integral(relupower + 0.01*abs(fddot),t)) #0.0001*relufddot,t)) # positive work 
# You get a smooth trajectory with both relu and abs, but solver has difficulty
#@objective(eyewft, Min, integral(relupower + 0.001*relufddot,t)) # positive work 

@constraint(eyewft, deriv(p,t) == v) # Dynamics
@constraint(eyewft, deriv(v,t) == f)
@constraint(eyewft, deriv(f,t) == fdot)
#@constraint(eyewft, deriv(fdot,t) == (fddotp - fddotm))
@constraint(eyewft, deriv(fdot,t) == fddot)
#@constraint(eyewft, f*v == powp + pown)

@constraint(eyewft, p(0) == 0) # Boundary conditions
@constraint(eyewft, p(1) == 1)
@constraint(eyewft, v(0) == 0)
@constraint(eyewft, v(1) == 0)
@constraint(eyewft, f(0) == 0)
@constraint(eyewft, f(1) == 0)
@constraint(eyewft, fdot(0) == 0)
@constraint(eyewft, fdot(1) == 0)

optimize!(eyewft)
plot(supports(t), value(v,ndarray=true))
plot(supports(t), value(fp).*value(v))
plot!(supports(t), value(fn).*value(v))
plot!(supports(t), value(relufddot))
plot(supports(t), value(f).*value(v))
pfig = plotpvu(t,p,v,f)
plot!(pfig[3], supports(t), -value(fddotm, ndarray=true))
# This is a decent bell shape, although optimizer had a tough time
plot(supports(t), value(fn).*value(v))


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