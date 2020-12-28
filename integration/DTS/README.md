An implementation of the MACKS dual timestepping algorithm from

"Optimization of one-parameter family of integration formulae for solving stiff chemical-kinetic ODEs"
Youhi Morii and Eiji Shima, Scientific Reports (2020), 10:21330

The idea with this algorithm is that we can write a reaction ODE as

dX/dt = a(t) - X / b(t)

If we take a timestep small enough that a and b are approximately constant,
the analytical solution to this equation is

X(t) = X_0 * e^(-t / b) + a * b  * (1 - e^(-t / b))

In the limit t -> infinity, the equilibrium state is X = a * b.

Now if we discretize the ODE using the explicit forward Euler approach, we would have

X^{n+1} = X^{n} + dt * (a - X^{n} / b)

and if we chose exactly dt = b, we would get the exact converged solution. So forward
Euler can work well as long as you limit dt to the timescale ~b, which can be very
short near NSE.

If we want to take a longer timestep (concerned only with accuracy requirements, not
stability requirements), then what we need to do is find the converged equilibrium
solution for that dt. This is the central insight of the above paper. Given a choice
for dt, we iterate to the find the converged solution X^{n+1} using Equation 22.
