import main as num_wf
import analytical_solution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap


#####################################################################################
#                                                                                   #
#           Script for generating Quantum Carpet space-time plots.                  #
#       Visualizes the probability density evolution over position and time.        #
#           Choices such as IC, potentials etc. are made in main.py                 #
#                                                                                   #
#                 Created by Markus Salomonsson and Joel Wittrin                    #
#                                                                                   #
#####################################################################################


plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
})
    
# Define nodes for the custom high-contrast colormap.
color_nodes = [
    (0.0, "black"), 
    (0.2, 'red'),
    (0.5, 'orange'),
    (1.0, "yellow"),
]

# Create the custom colormap object.
my_cmap = LinearSegmentedColormap.from_list("custom_high_contrast", color_nodes)

def plot_QC_rad(dt = 0.0005, t0=0, ts =0.01, L=1, mx=1001, xl=0, xr=1, q=50):

    # Initialize wavefunction with radiation/absorbing boundary conditions.

    wf = num_wf.wavefunction_rad(mx=mx, xl=xl, xr=xr, x0=0.5, q=q)
    
    # Calculate the space-time evolution matrix (Quantum Carpet).
    QCM = wf.quantum_carpet_matrix(ts=ts)
    font_size = 16

    # Create figure and plot the quantum carpet density map.
    plt.figure('radiation', figsize=(5.5, 4))
    plt.imshow(
        QCM,
        aspect='auto',
        origin='lower',
        extent=[xl, xr, 0, ts],
        cmap=my_cmap,
        vmin=0, vmax=15,
        interpolation='nearest'
    )
    
    # Configure colorbar labels and dimensions.
    cbar = plt.colorbar()
    cbar.set_label(r'$|\psi|^2$', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    
    # Set axis titles and layout parameters.
    plt.xlabel('Position x', fontsize=font_size)
    plt.ylabel('Time t', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.tight_layout()
    return None

def plot_QC(dt = 0.000005, t0=0, ts =0.085, L=1, mx=1001, xl=0, xr=1):

    # Determine the width of the spatial grid domain.
    L = (xr - xl)
    
    # Initialize wavefunction with default hard boundary conditions.
    wf = num_wf.wavefunction(mx=mx, xl=xl, xr=xr, order = 6, x0=0.5)
    
    # Calculate the space-time evolution matrix.
    QCM = wf.quantum_carpet_matrix(ts=ts, t0=t0)
    font_size = 16
    
    # Create figure and plot the QC.
    plt.figure('neumann', figsize=(5.5, 4))
    plt.imshow(
        QCM,
        aspect='auto',
        origin='lower',
        extent=[xl, xr, t0, ts],
        cmap=my_cmap,
        vmin=0, vmax=np.max(QCM),
        interpolation='nearest'
    )
    
    # Configure colorbar labels and dimensions.
    cbar = plt.colorbar()
    cbar.set_label(r'$|\psi|^2$', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    
    # Set axis titles and layout parameters.
    plt.xlabel('Position x', fontsize=font_size)
    plt.ylabel('Time t', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.tight_layout()
    return None


# ---------------Only execute one QC at a time!----------------- #

#plot_QC()
plot_QC_rad()
plt.show()