**Computation of Leaky Quantum Carpets**

A numerical solver for the time-dependent 1D Schrödinger equation, supporting multiple boundary conditions (Dirichlet, Neumann, Sommerfeld/radiating) and potentials. The numerical solution can be compared against an analytical solution, and visualized as "quantum carpets" - space-time plots of the probability density.

Files of interest:

1. main.py - The computational core of the project. All parameters (initial conditions, potentials, boundary conditions, etc.) are chosen here.
2. animating_scripts.py - Animates the numerical solution alongside the analytical solution, based on the parameters set in main.py.
3. quantum_carpets.py - Generates quantum carpet plots (probability density over position and time), based on the parameters set in main.py.

These three files let you run and explore the project without writing any code yourself. All code has been used for the bachelor's project "Computation of Leaky Quantum Carpets".
