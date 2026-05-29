import main as num_wf
import analytical_solution
import numpy as np
import matplotlib.pyplot as plt


#####################################################################################
#                                                                                   #
#  Animation script for Neumann, Dirichlet and Sommerfeld BC. Choices such as IC,   #
#                    potentials etc. are made in main.py                            #
#                                                                                   #
#               Created by Markus Salomonsson & Joel Wittrin                        #
#                                                                                   #
#####################################################################################


def animating_dirichlet(dt = 0.0005, t0=0, ts =0.2, L=1, mx=101):
    wf = num_wf.wavefunction(
        bcs=num_wf.boundary_conditions(
            left_BC  = num_wf.BoundaryCondition('dirichlet', tau=1),
            right_BC = num_wf.BoundaryCondition('dirichlet', tau=-1),
        ), xr=L, x0=L/2, mx=mx, order=6
    )

    plt.ion()
    fig, ax = plt.subplots()

    for t in np.arange(t0, ts, dt):
        ax.clear()
        ax.set_ylim(0, 8) 
        wf.time_evolution(t=t)
        x, y = wf.plot_probability()
        ax.plot(x, y, label='Numerisk', c='blue')
        ax.plot(wf.gridpoints, np.abs(analytical_solution.analytical_solution_dirichlet(x0=L/2, xr=L, t=t, mx=mx))**2, label='Analytisk', ls='--', c='r')
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        plt.pause(0.2)
        
    plt.ioff()
    plt.show()


def animating_neumann(dt = 0.0005, t0=0, ts =0.2, L=1, mx=101):
    wf = num_wf.wavefunction(
        num_wf.boundary_conditions(
            left_BC  = num_wf.BoundaryCondition('neumann', tau=-1000),
            right_BC = num_wf.BoundaryCondition('neumann', tau=-1000),
        ), xr=L, x0=L/2, mx=mx
    )
    plt.ion()
    fig, ax = plt.subplots()

    for t in np.arange(t0, ts, dt):
        ax.clear()
        wf.time_evolution(t=t)
        x, y = wf.plot_probability()
        ax.plot(x, y, label='Numerisk')
        ax.plot(wf.gridpoints, np.abs(analytical_solution.analytical_solution_nuemann(x0=L/2, xr=L, t=t, mx=mx))**2, label='Analytisk')
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        plt.pause(0.2)

    plt.ioff()
    plt.show()


def animating_radiation(dt = 0.0001, t0=0, ts =0.1, L=1, mx=601):
    wf = num_wf.wavefunction_rad(xr=L, x0=0.5, mx=mx, order=6)
    wf.plot_potential()

    x, y = wf.plot_probability()
    y_max = 10

    plt.ion()
    fig, ax = plt.subplots()

    for t in np.arange(t0, ts, dt):
        ax.clear()
        wf.get_state_at_time(t=t)
        x, y = wf.plot_probability()
        ax.plot(x, y, label='Numerisk')
        ax.set_ylim(0, y_max * 1.1) 
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        plt.pause(0.1)

    
# ---------------DO NOT RUN MORE THAN ONE ANIMATION AT A TIME!----------------- #

#animating_dirichlet()
#animating_neumann()
animating_radiation()

# ----------------------------------------------------------------------------- #
