ERENA algorithm from

"ERENA: A fast and robust Jacobian-free integration method for
ordinary differential equations of chemical kinetics"
Morii et al., JCP (2016) 322, 547

The RENA algorithm in section 2.4 is a direct implementation of the
QSS analytical solution with renormalization of mass fractions. The
Extended RENA algorithm in section 2.5 modifies this to take a smaller
timestep if the sum of X is sufficiently differently from 1.
