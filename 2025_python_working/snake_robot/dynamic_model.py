"""
Snake Robot Dynamic Model Module
================================
Implements the snake robot dynamics including:
- Reference angle generation
- Control law (PD controller)
- ODE solver for state space model
- Friction and constraint forces

OPTIMIZED VERSION:
- Pre-allocated buffers for all intermediate computations
- Contiguous C-order arrays for cache efficiency
- Minimized memory allocations in ODE function
- Vectorized operations where possible
"""

import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import solve as np_solve  # Faster than inv @ for solving linear systems
from tqdm import tqdm

try:
    from .config import SIMULATION_SETTINGS, STEERING_CONFIG
except ImportError:
    from config import SIMULATION_SETTINGS, STEERING_CONFIG


class SnakeDynamicModel:
    """
    Snake robot dynamic model and simulation.
    """
    
    def __init__(self, phy_properties, initial_values, snake_parameters, control_gains):
        """
        Initialize the dynamic model.
        
        Args:
            phy_properties: Physical properties dictionary
            initial_values: Initial state values dictionary
            snake_parameters: Motion parameters dictionary
            control_gains: Control gains dictionary
        """
        self.snake_parameters = snake_parameters
        
        # Physical parameters
        self.l = phy_properties["l"]
        self.m = phy_properties["m"]
        self.N = phy_properties["N"]
        self.J = (self.m * self.l ** 2) / 3  # Moment of inertia
        self.g = phy_properties["g"]
        self.c_n = phy_properties["c_n"]
        self.c_t = phy_properties["c_t"]
        
        # Pre-compute friction difference (used frequently)
        self.c_diff = self.c_t - self.c_n
        
        # Control gains
        self.k_p = control_gains["kp"]
        self.k_d = control_gains["kd"]
        self.k_i = control_gains["ki"]
        
        # Time span
        self.T_span = snake_parameters["T_span"]
        
        # Build constant matrices (contiguous, C-order)
        self._build_matrices()
        
        # Pre-allocate buffers for ODE computations
        self._allocate_buffers()
        
        # Initialize state from initial values
        self._init_state(initial_values)
        
        # Solver settings
        self.solver_method = SIMULATION_SETTINGS.get("solver_method", "RK45")
        self.rtol = SIMULATION_SETTINGS.get("rtol", 1e-6)
        self.atol = SIMULATION_SETTINGS.get("atol", 1e-8)
        
    def _build_matrices(self):
        """Build constant matrices used in dynamics. All matrices are contiguous C-order for speed."""
        N = self.N
        
        # A matrix: R^(N-1 x N) - contiguous
        A = np.eye(N - 1, N, dtype=np.float64) + np.hstack([
            np.zeros((N - 1, 1), dtype=np.float64),
            np.eye(N - 1, dtype=np.float64)
        ])
        self.A = np.ascontiguousarray(A)
        
        # D matrix: R^(N-1 x N) - contiguous
        D = np.eye(N - 1, N, dtype=np.float64) + np.hstack([
            np.zeros((N - 1, 1), dtype=np.float64),
            -np.eye(N - 1, dtype=np.float64)
        ])
        self.D = np.ascontiguousarray(D)
        
        # e vector: R^(N,) as 1D for speed
        self.e_flat = np.ones(N, dtype=np.float64)
        self.e = self.e_flat.reshape(-1, 1)  # Keep 2D version for compatibility
        
        # ones vector (N,) for matrix operations
        self.ones_N = np.ones(N, dtype=np.float64)
        
        # H matrix for theta = H * phi_bar transformation - contiguous
        H = -np.triu(np.ones((N, N), dtype=np.float64))
        H[:, -1] = -H[:, -1]
        self.H = np.ascontiguousarray(H)
        self.H_T = np.ascontiguousarray(H.T)  # Pre-compute transpose
        
        # V and K matrices - contiguous
        DDT_inv = np.linalg.inv(np.dot(self.D, self.D.T))
        self.V = np.ascontiguousarray(np.dot(np.dot(self.A.T, DDT_inv), self.A))
        self.K = np.ascontiguousarray(np.dot(np.dot(self.A.T, DDT_inv), self.D))
        self.K_T = np.ascontiguousarray(self.K.T)  # Pre-compute transpose
        
        # Pre-compute l*K_T for velocity calculations
        self.l_K_T = self.l * self.K_T
        
        # Pre-compute mass matrix constant term
        self.J_eye = self.J * np.eye(N, dtype=np.float64)
        self.ml2 = self.m * self.l**2
        self.Nm_eye2 = N * self.m * np.eye(2, dtype=np.float64)
        
    def _allocate_buffers(self):
        """Pre-allocate reusable buffers for ODE computations to minimize allocations."""
        N = self.N
        
        # State extraction buffers
        self._phi_bar = np.zeros(N, dtype=np.float64)
        self._phi_bar_d = np.zeros(N, dtype=np.float64)
        self._theta = np.zeros(N, dtype=np.float64)
        self._theta_d = np.zeros(N, dtype=np.float64)
        
        # Trig function buffers
        self._sin_theta = np.zeros(N, dtype=np.float64)
        self._cos_theta = np.zeros(N, dtype=np.float64)
        self._sin_phi = np.zeros(N, dtype=np.float64)
        self._cos_phi = np.zeros(N, dtype=np.float64)
        
        # Velocity buffers
        self._X_d = np.zeros(N, dtype=np.float64)
        self._Y_d = np.zeros(N, dtype=np.float64)
        
        # Friction buffers
        self._F_rv = np.zeros(2*N, dtype=np.float64)
        self._vel = np.zeros(2*N, dtype=np.float64)
        
        # Dynamics matrix buffers - pre-allocate ALL intermediate matrices
        self._M_theta = np.zeros((N, N), dtype=np.float64)
        self._M_11 = np.zeros((N, N), dtype=np.float64)
        self._M_phib = np.zeros((N+2, N+2), dtype=np.float64)
        self._W_mat = np.zeros((N, N), dtype=np.float64)
        self._W_phib = np.zeros((N+2, 1), dtype=np.float64)
        self._G_phib = np.zeros((N+2, 2*N), dtype=np.float64)
        
        # Friction matrix components (diagonal, so store as vectors)
        self._F11_diag = np.zeros(N, dtype=np.float64)
        self._F12_diag = np.zeros(N, dtype=np.float64)
        self._F22_diag = np.zeros(N, dtype=np.float64)
        
        # Control buffers
        self._u_bar = np.zeros(N-1, dtype=np.float64)
        self._phir = np.zeros(N-1, dtype=np.float64)
        self._phir_d = np.zeros(N-1, dtype=np.float64)
        self._phir_dd = np.zeros(N-1, dtype=np.float64)
        
        # Output buffer
        self._y_dot = np.zeros(2*(N+2), dtype=np.float64)
        
        # Temp buffers for matrix operations
        self._temp_N = np.zeros(N, dtype=np.float64)
        self._temp_N2 = np.zeros(N, dtype=np.float64)
        self._temp_NxN = np.zeros((N, N), dtype=np.float64)
        self._temp_Nx1 = np.zeros((N, 1), dtype=np.float64)
        
        # Pre-allocate fixed zero matrices that don't change
        self._M_12 = np.zeros((N, 2), dtype=np.float64)
        self._M_21 = np.zeros((2, N), dtype=np.float64)
        self._zeros_2x1 = np.zeros((2, 1), dtype=np.float64)
        
    def _init_state(self, initial_values):
        """Initialize state variables from initial values."""
        # Actuated coordinates: qa = [phi1, ..., phi_{N-1}]
        self.x1 = np.asarray(initial_values["qa_phi"]).flatten()
        
        # Unactuated coordinates: qu = [theta_N, px, py]
        self.x2 = np.asarray(initial_values["qu"]).flatten()
        
        # Actuated velocities
        self.x3 = np.asarray(initial_values["qa_phi_d"]).flatten()
        
        # Unactuated velocities
        self.x4 = np.asarray(initial_values["qu_d"]).flatten()
        
        # Full state vector q = [x1; x2; x3; x4] as 1D array
        self.q = np.concatenate([self.x1, self.x2, self.x3, self.x4])
        
        # Accelerations
        self.qa_ddot = np.asarray(initial_values["qa_phi_dd"]).flatten()
        self.qu_ddot = np.asarray(initial_values["qu_dd"]).flatten()
        
    def generate_reference_angles(self):
        """
        Generate reference joint angles for the entire simulation.
        
        Returns:
            tuple: (j_angle, j_angle_d, j_angle_dd, T_span) dictionaries and time array
        """
        params = self.snake_parameters
        a = params["alpha"]
        w = params["freq_w"]
        d = params["delta"]
        T_span = params["T_span"]
        
        j_angle = {}
        j_angle_d = {}
        j_angle_dd = {}
        
        print("Generating reference angles...")
        for i in tqdm(range(self.N - 1), desc="Joints"):
            angles = np.zeros(len(T_span))
            angles_d = np.zeros(len(T_span))
            angles_dd = np.zeros(len(T_span))
            
            for j, t in enumerate(T_span):
                phi0 = self._get_steering_offset(t)
                angles[j] = a * np.sin(w * t + i * d) + phi0
                angles_d[j] = a * w * np.cos(w * t + i * d)
                angles_dd[j] = -a * w**2 * np.sin(w * t + i * d)
                
            j_angle[f"phi{i+1}"] = angles
            j_angle_d[f"phi_d{i+1}"] = angles_d
            j_angle_dd[f"phi_dd{i+1}"] = angles_dd
            
        return j_angle, j_angle_d, j_angle_dd, T_span
    
    def _get_steering_offset(self, t):
        """
        Get time-dependent steering offset for turning.
        
        Args:
            t: Current time
            
        Returns:
            float: Joint offset in radians
        """
        for turn_name, turn_config in STEERING_CONFIG.items():
            if turn_config["t_start"] <= t <= turn_config["t_end"]:
                return turn_config["offset"]
        return 0.0
    
    def reference_angles_at_time(self, t):
        """
        Compute reference angles at a specific time t.
        
        Args:
            t: Time instant
            
        Returns:
            tuple: (angles, angles_d, angles_dd) as numpy arrays
        """
        params = self.snake_parameters
        a = params["alpha"]
        w = params["freq_w"]
        d = params["delta"]
        
        phi0 = self._get_steering_offset(t)
        
        angles = np.zeros(self.N - 1)
        angles_d = np.zeros(self.N - 1)
        angles_dd = np.zeros(self.N - 1)
        
        for i in range(self.N - 1):
            angles[i] = a * np.sin(w * t + i * d) + phi0
            angles_d[i] = a * w * np.cos(w * t + i * d)
            angles_dd[i] = -a * w**2 * np.sin(w * t + i * d)
            
        return angles, angles_d, angles_dd
    
    def control_law(self, phi, phi_d, phir, phir_d, phir_dd):
        """
        PD control law for joint angle tracking.
        
        Args:
            phi: Current joint angles
            phi_d: Current joint angular velocities
            phir: Reference joint angles
            phir_d: Reference joint angular velocities
            phir_dd: Reference joint angular accelerations
            
        Returns:
            numpy.ndarray: Control input u_bar
        """
        u_bar = phir_dd + self.k_d * (phir_d - phi_d) + self.k_p * (phir - phi)
        return u_bar
    
    def simulate(self, verbose=True):
        """
        Run the dynamic simulation with progress display.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            tuple: (time_array, state_array)
        """
        import time as time_module
        import sys
        
        t_end = self.T_span[-1]
        t_start = self.T_span[0]
        
        if verbose:
            print(f"\nStarting simulation...")
            print(f"  Method: {self.solver_method}")
            print(f"  Time span: {t_start} to {t_end} s")
            print(f"  Tolerances: rtol={self.rtol}, atol={self.atol}")
            print(f"\nProgress: ", end="", flush=True)
        
        # Progress tracking variables
        self._last_progress_time = time_module.time()
        self._last_progress_t = t_start
        self._progress_interval = 0.5  # Update every 0.5 seconds of real time
        self._verbose = verbose
        self._t_end = t_end
        self._t_start = t_start
        self._sim_start_time = time_module.time()
        
        # Initial conditions (ensure 1D array)
        y0 = np.asarray(self.q).flatten()
        t_span = [t_start, t_end]
        
        # Wrapper function that includes progress reporting
        def derivatives_with_progress(t, y):
            # Check if we should update progress (every 0.5s real time)
            current_time = time_module.time()
            if self._verbose and (current_time - self._last_progress_time) >= self._progress_interval:
                elapsed = current_time - self._sim_start_time
                progress_pct = (t - self._t_start) / (self._t_end - self._t_start) * 100
                # Estimate remaining time
                if progress_pct > 0:
                    eta = elapsed / progress_pct * (100 - progress_pct)
                    print(f"\r  Sim time: {t:.1f}/{self._t_end:.1f}s ({progress_pct:.0f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s   ", end="", flush=True)
                else:
                    print(f"\r  Sim time: {t:.1f}/{self._t_end:.1f}s ({progress_pct:.0f}%) | Elapsed: {elapsed:.1f}s   ", end="", flush=True)
                self._last_progress_time = current_time
                self._last_progress_t = t
            
            return self._compute_derivatives(t, y)
        
        # Solve ODE
        sol = solve_ivp(
            derivatives_with_progress,
            t_span,
            y0,
            method=self.solver_method,
            t_eval=self.T_span,
            rtol=self.rtol,
            atol=self.atol
        )
        
        if verbose:
            total_time = time_module.time() - self._sim_start_time
            print(f"\r  Sim time: {t_end:.1f}/{t_end:.1f}s (100%) | Elapsed: {total_time:.1f}s | Done!        ")
            print(f"\n  Simulation {'successful' if sol.success else 'FAILED'}")
            print(f"  Function evaluations: {sol.nfev}")
            print(f"  Real-time factor: {t_end/total_time:.1f}x")
        
        self.t_result = sol.t
        self.y_result = sol.y
        
        return sol.t, sol.y
    
    def _compute_derivatives(self, t, y):
        """
        Compute state derivatives for the ODE solver.
        FULLY OPTIMIZED: Uses pre-allocated buffers, avoids np.diag(), uses np.linalg.solve().
        
        Args:
            t: Current time
            y: Current state vector (1D array)
            
        Returns:
            numpy.ndarray: State derivatives (1D array)
        """
        N = self.N
        Nm1 = N - 1
        
        # Extract state components (1D array slicing - fast, no copy)
        x1 = y[:Nm1]              # Joint angles [phi1, ..., phi_{N-1}]
        x2 = y[Nm1:N+2]           # [theta_N, px, py]
        x3 = y[N+2:2*N+1]         # Joint velocities
        x4 = y[2*N+1:2*N+4]       # [theta_N_d, px_d, py_d]
        
        # Build phi_bar using pre-allocated buffer
        self._phi_bar[:Nm1] = x1
        self._phi_bar[Nm1] = x2[0]
        
        self._phi_bar_d[:Nm1] = x3
        self._phi_bar_d[Nm1] = x4[0]
        
        # Compute link angles using pre-allocated buffers
        np.dot(self.H, self._phi_bar, out=self._theta)
        np.dot(self.H, self._phi_bar_d, out=self._theta_d)
        
        # Trig values using pre-allocated buffers
        np.sin(self._theta, out=self._sin_theta)
        np.cos(self._theta, out=self._cos_theta)
        np.sin(self._phi_bar, out=self._sin_phi)
        np.cos(self._phi_bar, out=self._cos_phi)
        
        # Shortcuts for trig vectors
        s = self._sin_theta
        c = self._cos_theta
        s_phi = self._sin_phi
        c_phi = self._cos_phi
        
        # Position and velocity (scalars)
        p_x_d, p_y_d = x4[1], x4[2]
        
        # Link CM velocities - FIXED to match MATLAB reference:
        # MATLAB: X_d_ode = l*K'*S_theta*theta_d_ode + e.*p_d(1)
        # MATLAB: Y_d_ode = -l*K'*C_theta*theta_d_ode + e.*p_d(2)
        # diag(s) @ theta_d = s * theta_d (element-wise)
        s_td = s * self._theta_d  # S_theta @ theta_d
        c_td = c * self._theta_d  # C_theta @ theta_d
        
        # X_d = l * K.T @ (s * theta_d) + p_x_d  (POSITIVE sign!)
        np.dot(self.l_K_T, s_td, out=self._X_d)
        self._X_d += p_x_d
        
        # Y_d = -l * K.T @ (c * theta_d) + p_y_d  (NEGATIVE sign!)
        np.dot(self.l_K_T, c_td, out=self._Y_d)
        self._Y_d *= -1
        self._Y_d += p_y_d
        
        # Anisotropic friction - OPTIMIZED: diagonal matrices, use element-wise ops
        # F11 = c_t * c² + c_n * s² (diagonal)
        # F12 = (c_t - c_n) * s * c (diagonal)
        # F22 = c_t * s² + c_n * c² (diagonal)
        c2 = c * c
        s2 = s * s
        sc = s * c
        
        self._F11_diag[:] = self.c_t * c2 + self.c_n * s2
        self._F12_diag[:] = self.c_diff * sc
        self._F22_diag[:] = self.c_t * s2 + self.c_n * c2
        
        # F_rv = -F @ [X_d; Y_d] where F is block diagonal
        # F_rv[:N] = -(F11 * X_d + F12 * Y_d)
        # F_rv[N:] = -(F12 * X_d + F22 * Y_d)
        self._F_rv[:N] = -(self._F11_diag * self._X_d + self._F12_diag * self._Y_d)
        self._F_rv[N:] = -(self._F12_diag * self._X_d + self._F22_diag * self._Y_d)
        
        # Mass matrix - OPTIMIZED: use outer product instead of diag @ matrix @ diag
        # S_phi @ V @ S_phi = s_phi[:, None] * V * s_phi[None, :]
        sv = s_phi[:, None] * self.V * s_phi[None, :]
        cv = c_phi[:, None] * self.V * c_phi[None, :]
        
        # M_theta = J*I + m*l² * (sv + cv)
        np.copyto(self._M_theta, self.J_eye)
        self._M_theta += self.ml2 * (sv + cv)
        
        # M_11 = H.T @ M_theta @ H
        np.dot(self._M_theta, self.H, out=self._temp_NxN)
        np.dot(self.H_T, self._temp_NxN, out=self._M_11)
        
        # Build M_phib (reuse pre-allocated, only update M_11 block)
        self._M_phib[:N, :N] = self._M_11
        self._M_phib[:N, N:N+2] = 0  # M_12 = 0
        self._M_phib[N:N+2, :N] = 0  # M_21 = 0
        self._M_phib[N, N] = N * self.m
        self._M_phib[N+1, N+1] = N * self.m
        
        # Coriolis/centrifugal W - OPTIMIZED
        # W_mat = m*l² * (S_phi @ V @ C_phi - C_phi @ V @ S_phi)
        scv = s_phi[:, None] * self.V * c_phi[None, :]
        csv = c_phi[:, None] * self.V * s_phi[None, :]
        np.subtract(scv, csv, out=self._W_mat)
        self._W_mat *= self.ml2
        
        # H_phi_d = H @ phi_bar_d = theta_d (reuse _temp_N)
        np.dot(self.H, self._phi_bar_d, out=self._temp_N)
        
        # W_term = H.T @ W_mat @ diag(H @ phi_bar_d) @ (H @ phi_bar_d)
        # MATLAB: Wb_phib_phibdot = [H'*W_phib*diag(H*phi_bar_d_ode)*H*phi_bar_d_ode; zeros(2,1)]
        # H @ phi_bar_d = theta_d, so: H.T @ W @ diag(theta_d) @ theta_d
        # diag(v) @ v = v * v element-wise = v²
        theta_d_sq = self._temp_N * self._temp_N  # theta_d² (element-wise)
        W_term = np.dot(self.H_T, np.dot(self._W_mat, theta_d_sq))
        
        self._W_phib[:N, 0] = W_term
        self._W_phib[N:, 0] = 0
        
        # Constraint Jacobian G - OPTIMIZED
        # G_11 = [-l * H.T @ diag(s) @ K, l * H.T @ diag(c) @ K]
        # diag(s) @ K = s[:, None] * K
        sK = s[:, None] * self.K
        cK = c[:, None] * self.K
        
        G_11_left = -self.l * np.dot(self.H_T, sK)
        G_11_right = self.l * np.dot(self.H_T, cK)
        
        self._G_phib[:N, :N] = G_11_left
        self._G_phib[:N, N:] = G_11_right
        self._G_phib[N, :N] = -1  # -e.T
        self._G_phib[N, N:] = 0
        self._G_phib[N+1, :N] = 0
        self._G_phib[N+1, N:] = -1  # -e.T
        
        # Partition for actuated/unactuated (views, no copy)
        M22 = self._M_phib[Nm1:N+2, Nm1:N+2]
        M21 = self._M_phib[Nm1:N+2, :Nm1]
        W2 = self._W_phib[Nm1:N+2]
        G2 = self._G_phib[Nm1:N+2, :]
        
        # Reference angles and control
        self._reference_angles_inplace(t)
        
        # Control law: u_bar = phir_dd + k_d * (phir_d - x3) + k_p * (phir - x1)
        np.subtract(self._phir_d, x3, out=self._u_bar)
        self._u_bar *= self.k_d
        self._u_bar += self._phir_dd
        self._u_bar += self.k_p * (self._phir - x1)
        
        # Solve for unactuated accelerations - USE np.linalg.solve (faster than inv)
        G2_F = np.dot(G2, self._F_rv)
        rhs = -(W2.flatten() + G2_F)
        
        # qu_ddot = M22^{-1} @ (rhs - M21 @ u_bar)
        # Use solve: M22 @ qu_ddot = rhs - M21 @ u_bar
        rhs_full = rhs - np.dot(M21, self._u_bar)
        qu_ddot = np.linalg.solve(M22, rhs_full)
        
        # Build output using pre-allocated buffer
        self._y_dot[:Nm1] = x3                    # phi_dot
        self._y_dot[Nm1:N+2] = x4                 # qu_dot
        self._y_dot[N+2:2*N+1] = self._u_bar      # phi_ddot (control)
        self._y_dot[2*N+1:2*N+4] = qu_ddot        # qu_ddot
        
        return self._y_dot.copy()  # Must copy to avoid aliasing
    
    def _reference_angles_inplace(self, t):
        """Compute reference angles at time t, storing in pre-allocated buffers."""
        a = self.snake_parameters["alpha"]
        w = self.snake_parameters["freq_w"]
        d = self.snake_parameters["delta"]
        
        phi0 = self._get_steering_offset(t)
        
        for i in range(self.N - 1):
            wt_id = w * t + i * d
            self._phir[i] = a * np.sin(wt_id) + phi0
            self._phir_d[i] = a * w * np.cos(wt_id)
            self._phir_dd[i] = -a * w * w * np.sin(wt_id)
    
    def get_results(self):
        """
        Get simulation results.
        
        Returns:
            tuple: (time, state) arrays
        """
        return self.t_result, self.y_result
    
    def extract_states(self, y=None):
        """
        Extract individual state components from result array.
        
        Args:
            y: State array (uses stored result if None)
            
        Returns:
            dict: Dictionary with extracted states
        """
        if y is None:
            y = self.y_result
            
        N = self.N
        return {
            'phi': y[0:N-1, :],           # Joint angles
            'theta_n': y[N-1, :],          # Head angle
            'px': y[N, :],                 # CM x position
            'py': y[N+1, :],               # CM y position
            'phi_d': y[N+2:2*N+1, :],      # Joint velocities
            'theta_n_d': y[2*N+1, :],      # Head angular velocity
            'px_d': y[2*N+2, :],           # CM x velocity
            'py_d': y[2*N+3, :]            # CM y velocity
        }


if __name__ == "__main__":
    # Test the dynamic model
    from snake_init import SnakeInitializer
    
    print("Testing Snake Dynamic Model...")
    
    # Initialize
    snake_init = SnakeInitializer()
    params = snake_init.get_parameters()
    
    # Create model
    model = SnakeDynamicModel(*params)
    
    # Generate reference angles
    ref_angles, _, _, T = model.generate_reference_angles()
    
    # Run simulation
    t, y = model.simulate()
    
    print(f"\nSimulation completed!")
    print(f"Time points: {len(t)}")
    print(f"State shape: {y.shape}")
