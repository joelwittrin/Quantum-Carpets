import numpy as np
import matplotlib.pyplot as plt
import operators as ops
import scipy
import potentials



################################################################################
#                                                                              #
#                            Quantum Carpet classes,                           #
#         used in "Computation of Leaky Quantum Carpets" created by            #
#                     Markus Salomonsson & Joel Wittrin                        #
#                                                                              #
#                                                                              #
#       This project utilizes a SBP-SAT method to numerically solve TDSE.      #
#             We encourage you to try out different potentials and ICs!        #
#                                                                              #
#      The classes that are crucial when changing the QCs are wavefunction     #
#  and wavefunction_rad, where all relevant parameters are set in __init__().  #
#                                                                              #
################################################################################



def IC_Gaussian(x, x0=0.5, l0=0.1, q=0):

    # A regular Gaussian curve.

    psi = np.exp(-((x - x0) / l0)**2)
    A0 = 1 / np.sqrt(np.trapz(np.abs(psi)**2, x))
    return (A0 * psi).astype(complex)

def IC_Gaussian_moving_right_and_left(x, x0=0.5, l0=0.1,q=20):

    # Identical to IC_Gaussian but with implemented momentum.

    psi = np.exp(-((x - x0) / l0)**2)
    A0 = 1 / np.sqrt(np.trapz(np.abs(psi)**2, x))
    return (A0 * psi).astype(complex)*np.cos(q*(x-x0))

def IC_Gaussian_moving_right(x, x0=0.5, l0=0.1,q=1):

    #A Gaussian curve, moving to the right.

    psi = np.exp(-((x - x0) / l0)**2)
    A0 = 1 / np.sqrt(np.trapz(np.abs(psi)**2, x))
    return (A0 * psi).astype(complex)*np.exp(1j*q*(x-x0))



class BoundaryCondition:

    # Container class for retaining the BC information.

    def __init__(self, bc_type, tau,k0=1):
        self.bc_type = bc_type
        self.tau     = tau
        self.k0      = k0

class boundary_conditions:

    # This class calculates the SAT-terms for Dirichlet, Neumann and Sommerfeld's radiation BC.

    def __init__(self, left_BC, right_BC):
        self.left_BC  = left_BC
        self.right_BC = right_BC

    def compute_sat_v2(self, HI, e_l, e_r, d1_l, d1_r):

        # Returns SAT-terms before multiplying with the state Psi.
        
        # Left BC
        if self.left_BC.bc_type == 'dirichlet':
            sat_left = self.left_BC.tau * HI @ (d1_l.T @ e_l)
            
        elif self.left_BC.bc_type == 'neumann':

            sat_left = self.left_BC.tau * HI @ e_l.T @ d1_l
        
        elif self.left_BC.bc_type == 'radiation':
            sat_left = self.left_BC.tau * HI @ e_l.T@(d1_l+1j*self.left_BC.k0*e_l)
        else:
            sat_left = 'error'

        # Right BC
        if self.right_BC.bc_type == 'dirichlet':
            sat_right = self.right_BC.tau * HI @ (d1_r.T @ e_r)
            
        elif self.right_BC.bc_type == 'neumann':
            sat_right = self.right_BC.tau * HI @ e_r.T @ d1_r

        elif self.right_BC.bc_type == 'radiation':
            sat_right = self.right_BC.tau * HI @ e_r.T@(d1_r-1j*self.left_BC.k0*e_r)

        else:
            sat_right = 'error'
        return sat_left + sat_right



class wavefunction:

    # Initializes the wavefunction (designed for Dirichlet and Neumann BCs).

    # Select an Initial Condition and one of the following potentials:
    # - zero_potential          - harmonic_potential       - constant_potential
    # - gaussian_well           - dirichlet_potential      - finite_well_potential
    # - potential_wall          - potential_wall_right     - double_well_potential


    def __init__(
        self,
        bcs = boundary_conditions(
            left_BC=BoundaryCondition('dirichlet', tau=1),      #CHOOSE FROM 'dirichlet', 'neumann' OR 'radiation'
            right_BC=BoundaryCondition('dirichlet', tau=-1),    #CHOOSE FROM 'dirichlet', 'neumann' OR 'radiation'
        ),

        v=potentials.dirichlet_potential,                       #CHOOSE POTENTIAL HERE.
        xl=0,
        xr=1,
        order=6,                                                #CHOOSE FROM ACCURACY ORDER 2, 4 OR 6.
        IC=IC_Gaussian,                                         #CHOOSE INITIAL CONDITION HERE.
        x0=0.5,
        mx=101,
        dt = 0.00001,
        q = 1
                ):

        self.hx = (xr - xl) / (mx - 1)

        if order     == 2:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_2nd(mx, self.hx)
        elif order   == 4:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_4th(mx, self.hx)
        elif order   == 6:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_6th(mx, self.hx)

        self.bcs        = bcs
        self.time       = 0
        self.dt         = dt
        self.gridpoints = np.linspace(xl, xr, mx)


        # V must be a diagonal matrix of size (mx x mx) containing the discrete potential.
        # This structure allows Psi to be factored out, enabling D2 and V to be added.
        self.V       = np.diag(v(self.gridpoints))

        # The initial condition sampled at the gridpoints
        self.state   = IC(self.gridpoints, x0=x0)

        # Attributes that are given values in solve_eigenvalue_problem
        self.cn0            = None
        self.cn             = None
        self.eigenvalues    = None
        self.eigenvectors   = None
        self.solve_eigenvalue_problem()


    def rhs_matrix(self):

        # Defines the operator matrix M for TISE,
        # expressed in the form: E * phi = M * phi, where M combines the right-hand side matrices.

        sat_terms = self.bcs.compute_sat_v2(self.HI, self.e_l, self.e_r, self.d1_l, self.d1_r)

        return -self.D2  +self.V  + sat_terms


    def solve_eigenvalue_problem(self):
            
            # Solves the TISE eigenvalue problem and projects the initial state

            # Matrix derivation for the eigenvalue problem E * phi = M * phi:
            # Continuous form: E*phi = -D2*phi + V*phi + HI*tau_l*e_l*(e_l.T*phi) + HI*tau_r*e_r*(e_r.T*phi)
            # Factored form:   E*phi = (-D2 + V + HI*tau_l*(e_l @ e_l.T) + HI*tau_r*(e_r @ e_r.T)) * phi

            # Construct the combined system matrix M and solve the eigenvalue problem
            M = self.rhs_matrix()
            self.eigenvalues, self.eigenvectors = np.linalg.eig(M)
            self.eigenvectors = np.asarray(self.eigenvectors)

            # Normalize eigenvectors with respect to the SBP norm
            for i in range(self.eigenvectors.shape[1]):
                vec = self.eigenvectors[:, i]
                self.eigenvectors[:, i] = vec / self._h_norm(vec)

            # Project initial state onto the eigenbasis using the SBP inner product
            self.cn0 = np.asarray(self.eigenvectors.T @ self.state).flatten()
            
            # Normalize expansion coefficients
            self.cn0 = self.cn0 / np.sqrt(np.sum(np.abs(self.cn0)**2))
            self.cn = self.cn0

            return None
    

    def plot_probability(self):

        # Returns the gridpoints and the probability density.

        return self.gridpoints, np.abs(np.asarray(self.state).flatten())**2


    def time_evolution(self, t, hbar=1):

        # Advances the state in time using the analytical solution to TDSE.

        self.cn=self.cn0*np.exp(-1j*self.eigenvalues*t/hbar)
        self.state = (self.eigenvectors @ self.cn)
        return None


    def total_prob(self):

        # Calculates the total probability using the SBP inner product norm.

        return np.trapz(np.abs(np.asarray(self.state).flatten())**2, self.gridpoints)
    

    def plot_potential(self):

        # Plot the chosen potential

        plt.figure('potential')
        plt.plot(self.gridpoints, np.diag(self.V))
        plt.show()


    def _h_norm(self, vec):

        # Returns the SBP H-norm sqrt(<vec|H|vec>) using sparse matrix operations.

        H = self.H.toarray()
        return np.sqrt(np.real(vec.conj() @ H @ vec))


    def quantum_carpet_matrix(self, ts, t0=0):

        # Generates the quantum carpet matrix by evolving the state over time.

        self.time =t0
        QCM = [np.abs(self.state)**2]
        while self.time <= ts:
            self.time += self.dt
            self.time_evolution(t=self.time) 
            QCM.append(np.abs(np.asarray(self.state).flatten())**2)
        QCM = np.array(QCM)
        return QCM
    

    def plot_eigenvectors(self):

        # Plots the most significant eigenvectors and prints their coefficients.

        # Plot and print for initial coefficients cn0
        print("=== INITIAL COEFFICIENTS (cn0) ===")
        self._create_eigenvector_plots(coefficients=self.cn0, threshold=0.001, label="cn0")
        print("\n" + "="*40 + "\n")

        # Plot and print for current coefficients cn
        print("=== CURRENT COEFFICIENTS (cn) ===")
        self._create_eigenvector_plots(coefficients=self.cn, threshold=0.00001, label="cn")


    def _create_eigenvector_plots(self, coefficients, threshold, label):

        # Function to handle the filtering, plotting, and printing logic.

        # Filter significant indices
        abs_coeffs = np.abs(coefficients)
        significant_indices = np.where(abs_coeffs > threshold)[0]
        print(f"Eigenvectors with |{label}| > {threshold}: {len(significant_indices)}")

        # Set up sub-plot grid dimensions
        n_vecs = min(12, len(significant_indices))
        if n_vecs > 0:
            n_cols = 4
            n_rows = int(np.ceil(n_vecs / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            # Plot each significant eigenvector
            for plot_idx in range(n_vecs):
                idx = significant_indices[plot_idx]
                ax = axes[plot_idx]
                
                vec = self.eigenvectors[:, idx]
                ax.plot(self.gridpoints, np.abs(vec)**2, linewidth=2)
                
                E = self.eigenvalues[idx]
                norm = np.linalg.norm(vec)
                cn_val = coefficients[idx]
                
                ax.set_title(f'Vec {idx}: E={E.real:.1f}+{E.imag:.1e}i, N={norm:.3f}\n|{label}|={abs_coeffs[idx]:.4f}', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for plot_idx in range(n_vecs, len(axes)):
                fig.delaxes(axes[plot_idx])

            plt.tight_layout()
            plt.show()

        # Print the top 20 largest coefficients sorted descending
        print(f"\nBiggest |{label}|-values:")
        sorted_indices = np.argsort(abs_coeffs)[::-1]
        for idx in sorted_indices[:20]:
            E = self.eigenvalues[idx]
            print(f"  Vec {idx}: |{label}|={abs_coeffs[idx]:.6f}, E={E.real:.2f}{E.imag:+.2e}i")



class wavefunction_rad:

    # Initializes the wavefunction (designed for Sommerfeld radiation BC).

    # Select an Initial Condition and one of the following potentials:
    # - zero_potential          - harmonic_potential       - constant_potential
    # - gaussian_well           - dirichlet_potential      - finite_well_potential
    # - potential_wall          - potential_wall_right     - double_well_potential

    def __init__(
        self,
        tau_r=1,
        tau_l=-1,
        v=potentials.potential_wall,            #CHOOSE POTENTIAL HERE.
        xl=0,
        xr=1,
        order=6,                                    #CHOOSE FROM ACCURACY ORDER 2, 4 OR 6.
        IC=IC_Gaussian_moving_right_and_left,       #CHOOSE INITIAL CONDITION HERE.
        x0=0.5,
        mx=1001,
        dt=0.00001,
        q=1,
    ):
        self.hx = (xr - xl) / (mx - 1)

        if order == 2:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_2nd(mx, self.hx)
        elif order == 4:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_4th(mx, self.hx)
        elif order == 6:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_6th(mx, self.hx)

        self.tau_r          = tau_r
        self.tau_l          = tau_l
        self.time           = 0
        self.dt             = dt
        self.mx             = mx
        self.gridpoints     = np.linspace(xl, xr, mx)
        self.initial_norm   = 1

        # See previous class for explained initialization choices.
        self.V = np.diag(v(self.gridpoints))
        self.state = IC(self.gridpoints, x0=x0, q=q)
        self.state /=self._h_norm(self.state)    
        self.cn = None
        
        # Solve for all times when initializing class
        self.solve_eigenvalue_problem()
        
    def plot_potential(self):

        # Plot the chosen potential

        plt.figure('potential')
        plt.plot(self.gridpoints, np.diag(self.V))
        plt.show()

    def _h_norm(self, vec):

        # Returns the SBP H-norm sqrt(<vec|H|vec>) using sparse matrix operations.

        H = self.H.toarray()
        return np.sqrt(np.real(vec.conj() @ H @ vec))
   
    def solve_eigenvalue_problem(self):
        
        # Solves the eigenvalue emerging from Sommerfeld, taking the complex energies into accout.

        CM = np.array(
            -self.D2  
            +self.V 
            +self.tau_l * self.HI @ self.e_l.T @ self.d1_l
            +self.tau_r * self.HI @ self.e_r.T @ self.d1_r
        )

        sqEM = (
            self.tau_l * self.HI @ self.e_l.T @ self.e_l * 1j
            -self.tau_r * self.HI @ self.e_r.T @ self.e_r * 1j
        ).toarray()

        A = np.block([[np.zeros((self.mx, self.mx)), np.eye(self.mx, dtype=complex)], [CM, sqEM]])
        
        # Eigenvalues (k) and eigenvectors
        self.k_list, self.eigvecs = scipy.linalg.eig(A)

        # Drop the auxillary variable
        self.eigvecs = self.eigvecs[:self.mx]

        # Time evolution: psi(t) ~ exp(-i * E * t)
        # With complex eigenvalues E = E_r + i * E_i:
        # exp(-i * (E_r + i * E_i) * t) = exp(-i * E_r * t) * exp(E_i * t)  
        self.all_energies = (self.k_list**2)[np.abs(np.imag(self.k_list**2))<1e4]
        stable_indices = (np.imag(self.k_list**2) < -1e-6) & (np.abs(np.imag(self.k_list**2)) < 1e4)

        self.k_stable = self.k_list[stable_indices]
        self.stable_energies = self.k_stable**2
        self.stable_eigenvectors = self.eigvecs[:, stable_indices]
        
        # Least square method as the eigenvectors do not form a orthogonal basis.
        self.cn = np.linalg.lstsq(self.stable_eigenvectors, self.state, rcond=None)[0]

    def get_state_at_time(self, t):

        # Evolves the wavefunction to time t analytically using the stable eigenbasis.

        time_factors = np.exp(-1j * self.stable_energies * t)
        self.state= self.stable_eigenvectors @ (time_factors * self.cn)
        return self.state
        
    def plot_probability(self):

        # Returns the gridpoints and the probability density.

        return self.gridpoints, np.abs(np.asarray(self.state).flatten())**2

    def total_prob(self):

        # Calculates the total probability using the SBP inner product norm.

        return np.trapz(np.abs(np.asarray(self.state).flatten())**2, self.gridpoints)
    
    def quantum_carpet_matrix(self, ts, t0=0):

        # Generates the quantum carpet matrix by evolving the state over time.

        self.time =t0
        QCM = [np.abs(np.asarray(self.state).flatten())**2]
        while self.time <= ts:
            self.time += self.dt
            self.get_state_at_time(self.time) 
            QCM.append(np.abs(np.asarray(self.state).flatten())**2)
        QCM = np.array(QCM)
        return QCM
    
    def plot_k_list_not_filtered(self):

        # Plots all unfiltered k-values in the complex plane.

        plt.figure('k not filtered')
        plt.scatter(self.k_list.real, self.k_list.imag, marker='o')
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.title('All k')
        plt.grid(True)

    def plot_k_list_filtered(self):

        # Plots only the stable, filtered k-values in the complex plane.

        plt.figure('k filtered')
        plt.scatter(self.k_stable.real, self.k_stable.imag, marker='o')
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.title('All stable k')
        plt.grid(True)
    
    def plot_eigenvals_not_filtered(self):

        # Plots all unfiltered eigenvalues in the complex plane.

        plt.scatter(self.all_energies.real, self.all_energies.imag, marker='o')
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.grid(True)

    def plot_common_energies(self):

        # Plots stable energies with an expansion coefficient magnitude above 0.01.

        plt.figure('Common energies')
        common_indicies = np.abs(self.cn) >0.01
        plt.scatter(self.stable_energies[common_indicies].real, self.stable_energies[common_indicies].imag, marker='o')
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.title('Energies with cn>0.01')
        plt.grid(True)

    def plot_eigenvals_filtered(self):

        # Plots only the stable, filtered eigenvalues.

        plt.figure('Eigenvalues filtered')
        plt.scatter(self.stable_energies.real, self.stable_energies.imag, marker='o')
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.title('Stable energies')
        plt.grid(True)

    def plot_eigenvectors(self):

        # Plots the probability density of stable eigenvectors with coefficients above 0.01.

        plt.figure('Eigenvectors')
        for i in range(self.num_stable):
            if self.cn[i]> 0.01: 
                plt.plot(self.gridpoints, np.abs(self.R_eigenvectors[:self.mx,i])**2)
        plt.xlabel('x')
        plt.ylabel(r'$|\phi_n|^2$')
        plt.title(r'Eigenvectors with $c_n>0.01$')
        plt.grid(True)

