# GPU-Master-Equation-Solver

The provided code (mesolve_gpu.py) can be used to simulate an open quantum system via the Lindblad Master Equation framework.  In particular, the code numerally solves the master equation (a PDE) in time, with discretized space, with the Runge-Kutta 4 scheme.

In order to use the provided code, the user need only call the runrk4_mesolve function.  Its inputs are:
1) The initial condition for the density matrix, which is called rho_initial
2) A list of times the user wants data collected, called tlist
3) The Hamiltonian of the system
4) The dissipation operators, which appear in the Lindblad Master Equation
5) The evaluation operators whose expectation values the user wants at the times listed in tlist
6) The coupling constants, which appear in the Lindblad Master Equation
7) OPTIONAL: whether or not the user wants an adaptive timestep (1 for adaptive timestep, 0 for fixed timestep).  This is 0 by default
8) OPTIONAL: the tolerance for the adaptive timestep.  This is 0.1 by default.
