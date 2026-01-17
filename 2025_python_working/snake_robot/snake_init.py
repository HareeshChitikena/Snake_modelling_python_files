"""
Snake Robot Initialization Module
==================================
Handles the initialization of the snake robot state including:
- Initial joint angles and velocities
- Initial link positions (center of mass)
- Coordinate transformations
"""

import numpy as np

try:
    from .config import PHYSICAL_PROPERTIES, INITIAL_CONDITIONS, CONTROL_GAINS, get_snake_parameters
except ImportError:
    from config import PHYSICAL_PROPERTIES, INITIAL_CONDITIONS, CONTROL_GAINS, get_snake_parameters


class SnakeInitializer:
    """
    Initialize snake robot state variables and compute initial conditions.
    """
    
    def __init__(self, phy_properties=None, initial_config=None, control_gains=None):
        """
        Initialize the snake robot with given or default parameters.
        
        Args:
            phy_properties: Physical properties dictionary (optional)
            initial_config: Initial conditions dictionary (optional)
            control_gains: Control gains dictionary (optional)
        """
        # Use provided or default configuration
        self.phy_properties = phy_properties or PHYSICAL_PROPERTIES.copy()
        self.initial_config = initial_config or INITIAL_CONDITIONS.copy()
        self.control_gains = control_gains or CONTROL_GAINS.copy()
        self.snake_parameters = get_snake_parameters()
        
        # Extract commonly used values
        self.N = self.phy_properties['N']
        self.l = self.phy_properties['l']
        
        # Initialize state variables
        self._init_state_variables()
        self._compute_initial_positions()
        self._build_initial_dictionary()
        
    def _init_state_variables(self):
        """Initialize joint angles, velocities, and accelerations."""
        # Joint angles qa_phi = [phi1, phi2, ..., phi_{N-1}] in R^(N-1 x 1)
        self.qa_phi = np.zeros([self.N - 1, 1])
        self.qa_phi[0] = self.initial_config.get('phi_1_0', np.deg2rad(20))
        
        # Joint angular velocities and accelerations
        self.qa_phi_d = np.zeros([self.N - 1, 1])
        self.qa_phi_dd = np.zeros([self.N - 1, 1])
        
        # Head link angle (theta_N)
        self.theta_n = np.matrix([[self.initial_config.get('theta_n_0', 0.0)]])
        self.theta_n_d = np.matrix([[0.0]])
        self.theta_n_dd = np.matrix([[0.0]])
        
        # Augmented angle vectors phi_bar = [phi1, ..., phi_{N-1}, theta_N]
        self.phi_bar = np.append(self.qa_phi, self.theta_n, axis=0)
        self.phi_bar_d = np.append(self.qa_phi_d, self.theta_n_d, axis=0)
        
        # Build H matrix for coordinate transformation (theta = H * phi_bar)
        self.H = -1 * np.triu(np.ones((self.N, self.N)))
        self.H[:, -1] = -1 * self.H[:, -1]
        
        # Compute link angles theta from phi_bar
        self.theta = np.matmul(self.H, self.phi_bar)
        self.theta_d = np.matmul(self.H, self.phi_bar_d)
        self.theta_bar = np.mean(self.theta)
        
    def _compute_initial_positions(self):
        """Compute initial center of mass positions for all links."""
        # Global coordinate origin
        X_0 = self.initial_config.get('X_0', np.array([0.0]))
        Y_0 = self.initial_config.get('Y_0', np.array([0.0]))
        starting_point = self.initial_config.get('starting_point', np.array([0, 0]))
        
        # Link 1 end point (tail)
        self.x1_i = X_0 + starting_point[0]
        self.y1_i = Y_0 + starting_point[1]
        
        # Link 1 center of mass
        self.x1g_t0 = self.x1_i + self.l * np.cos(float(self.theta[0]))
        self.y1g_t0 = self.y1_i + self.l * np.sin(float(self.theta[0]))
        
        # Initialize arrays for all link CM positions
        self.xg_t0 = np.array([float(self.x1g_t0)])
        self.yg_t0 = np.array([float(self.y1g_t0)])
        
        # Compute CM positions for links 2 to N
        for i in range(1, len(self.theta)):
            x_prev = self.xg_t0[i - 1]
            y_prev = self.yg_t0[i - 1]
            t1 = float(self.theta[i - 1])
            t2 = float(self.theta[i])
            x_new = x_prev + self.l * (np.cos(t1) + np.cos(t2))
            y_new = y_prev + self.l * (np.sin(t1) + np.sin(t2))
            self.xg_t0 = np.append(self.xg_t0, x_new)
            self.yg_t0 = np.append(self.yg_t0, y_new)
        
        # Initial CM velocities and accelerations (zero at start)
        self.xg_t0_d = np.zeros(self.N)
        self.yg_t0_d = np.zeros(self.N)
        self.xg_t0_dd = np.zeros(self.N)
        self.yg_t0_dd = np.zeros(self.N)
        
        # Snake robot overall center of mass
        self.px_t0 = np.mean(self.xg_t0)
        self.py_t0 = np.mean(self.yg_t0)
        self.p_t0 = np.array([[self.px_t0], [self.py_t0]])
        
        # CM velocities and accelerations
        self.px_t0_d = np.mean(self.xg_t0_d)
        self.py_t0_d = np.mean(self.yg_t0_d)
        self.p_t0_d = np.array([[self.px_t0_d], [self.py_t0_d]])
        
        self.px_t0_dd = np.mean(self.xg_t0_dd)
        self.py_t0_dd = np.mean(self.yg_t0_dd)
        self.p_t0_dd = np.array([[self.px_t0_dd], [self.py_t0_dd]])
        
        # Unactuated coordinates qu = [theta_N, px, py]
        self.qu = np.vstack([self.theta_n, self.p_t0])
        self.qu_d = np.vstack([self.theta_n_d, self.p_t0_d])
        self.qu_dd = np.vstack([self.theta_n_dd, self.p_t0_dd])
        
    def _build_initial_dictionary(self):
        """Build dictionary containing all initial values."""
        self.initial_values = {
            'x1_i': self.x1_i,
            'y1_i': self.y1_i,
            'qa_phi': self.qa_phi,
            'qa_phi_d': self.qa_phi_d,
            'qa_phi_dd': self.qa_phi_dd,
            'theta_n': self.theta_n,
            'theta_n_d': self.theta_n_d,
            'phi_bar': self.phi_bar,
            'phi_bar_d': self.phi_bar_d,
            'theta': self.theta,
            'theta_d': self.theta_d,
            'theta_bar': self.theta_bar,
            'xg_t0': self.xg_t0,
            'yg_t0': self.yg_t0,
            'xg_t0_d': self.xg_t0_d,
            'yg_t0_d': self.yg_t0_d,
            'px_t0': self.px_t0,
            'py_t0': self.py_t0,
            'px_t0_d': self.px_t0_d,
            'py_t0_d': self.py_t0_d,
            'qu': self.qu,
            'qu_d': self.qu_d,
            'qu_dd': self.qu_dd
        }
        
    def get_parameters(self):
        """
        Get all parameters needed for the dynamic model.
        
        Returns:
            tuple: (phy_properties, initial_values, snake_parameters, control_gains)
        """
        return (
            self.phy_properties,
            self.initial_values,
            self.snake_parameters,
            self.control_gains
        )
    
    def print_summary(self):
        """Print a summary of the initialization."""
        print("\n" + "=" * 60)
        print("Snake Robot Initialization Summary")
        print("=" * 60)
        print(f"Number of links: {self.N}")
        print(f"Link length: {self.l * 2:.4f} m (half-length: {self.l:.4f} m)")
        print(f"Initial head angle (theta_N): {np.rad2deg(float(self.theta_n)):.2f} deg")
        print(f"Initial snake CM position: ({self.px_t0:.4f}, {self.py_t0:.4f}) m")
        print(f"Control gains - Kp: {self.control_gains['kp']}, Kd: {self.control_gains['kd']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test the initializer
    snake = SnakeInitializer()
    snake.print_summary()
    params = snake.get_parameters()
    print("Parameters retrieved successfully!")
