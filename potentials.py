import numpy as np

#PLEASE NOTE THAT THIS FILE IS NOT ADAPTED FOR EXTERNAL EDUCATIONAL VIEWING.

def zero_potential(x):
    return x*0

def harmonic_potential(x):
    x0 = (x[-1]+x[0])/2
    return 1e3*(x-x0)**2

def constant_potential(x, v0=100):
    return np.ones_like(x)*v0

def gaussian_well(x):
    return 100 * 1-(np.exp(-100 * (x - 0.5)**2))

def dirichlet_potential(x, k=5000):
    V0 = 1e10 
    x_left = 0
    x_right = 1
    
    # Vänster vägg: Går från V0 till 0 vid x = 0
    left_wall = (V0 / 2) * (1 - np.tanh(k * (x - x_left)))
    
    # Höger vägg: Går från 0 till V0 vid x = 1
    right_wall = (V0 / 2) * (1 + np.tanh(k * (x - x_right)))
    
    return left_wall + right_wall

def finite_well_potential(x):
    V0 = 500 
    x_left = 0.1  # Vänster vägg
    x_right = 0.9 # Höger vägg    
    return np.where((x > x_left) & (x < x_right), 0.0, V0)

def potential_wall(x):
    # V_left = 1e10
    # x_left = 0.05
    
    V_barrier = 700
    x_barrier_start = 0.7
    x_barrier_end = 0.8
    
    #Start with small non-zero potential is needed for some reason
    V = np.ones_like(x)*10
    
    # V[x < x_left] = V_left
    
    # Finite barrier wall
    V[(x >= x_barrier_start) & (x <= x_barrier_end)] = V_barrier
    
    
    return V

def potential_wall_right(x):
    V_left = 1e10
    x_left = 0.05
    
    V_right = 900
    x_right = 0.9
    
   
    V = np.zeros_like(x)
    V[x < x_left] = V_left
    V[x >= x_right] = V_right
    return V

def double_well_potential(x, V0=1500.0, d=0.2):
    """
    V0: Höjden på barriären i mitten (vid x = 0.5).
    d:  Avståndet från mitten till botten av brunnarna. 
        Med d=0.2 hamnar brunnarna vid x=0.3 och x=0.7.
    """
    x_c = 0.5
    
    
    return V0 * (((x - x_c) / d)**2 - 1)**2