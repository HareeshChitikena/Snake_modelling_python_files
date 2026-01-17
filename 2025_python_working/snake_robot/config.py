"""
Snake Robot Configuration File
==============================
Contains all configurable parameters for the snake robot simulation.

Parameters:
- Physical properties (link length, mass, friction coefficients)
- Snake motion parameters (amplitude, frequency, phase offset)
- Control gains (PD controller)
- Simulation settings (time span, solver tolerances)
"""

import numpy as np


# =============================================================================
# DEFAULT VALUES (can be overridden by user input)
# =============================================================================
DEFAULT_NUM_LINKS = 10
DEFAULT_SIMULATION_TIME = 60.0


# =============================================================================
# PHYSICAL PROPERTIES
# =============================================================================
PHYSICAL_PROPERTIES = {
    "l": 0.14 / 2,          # Half-link length in meters
    "m": 1.0,               # Mass per link in kg
    "N": DEFAULT_NUM_LINKS, # Number of links (default, can be overridden)
    "g": 9.8,               # Gravity in m/s^2
    "c_n": 10.0,            # Viscous normal friction coefficient
    "c_t": 1.0,             # Viscous tangential friction coefficient
    "mu_n": 0.0,            # Coulomb normal friction coefficient
    "mu_t": 0.0             # Coulomb tangential friction coefficient
}


# =============================================================================
# SNAKE SINUSOIDAL MOTION PARAMETERS
# =============================================================================
SNAKE_MOTION_PARAMS = {
    "alpha": np.deg2rad(30),       # Amplitude in radians
    "freq_w": np.deg2rad(50),      # Angular frequency in rad/s
    "delta": np.deg2rad(40),       # Phase offset between joints in radians
    "joint_offset": np.deg2rad(0), # Initial joint offset in radians
}


# =============================================================================
# ACCURACY PRESETS
# =============================================================================
ACCURACY_PRESETS = {
    "1": {
        "name": "low",
        "rtol": 1e-2,
        "atol": 1e-2,
        "description": "Low accuracy (fast, rtol=1e-2, atol=1e-2)"
    },
    "2": {
        "name": "medium",
        "rtol": 1e-4,
        "atol": 1e-4,
        "description": "Medium accuracy (balanced, rtol=1e-4, atol=1e-4)"
    },
    "3": {
        "name": "high",
        "rtol": 1e-6,
        "atol": 1e-6,
        "description": "High accuracy (slow, rtol=1e-6, atol=1e-6)"
    }
}

DEFAULT_ACCURACY = "1"  # 1=low (default), 2=medium, 3=high


# =============================================================================
# SIMULATION SETTINGS
# =============================================================================
SIMULATION_SETTINGS = {
    "t_start": 0.0,                     # Start time in seconds
    "t_end": DEFAULT_SIMULATION_TIME,   # End time in seconds (default, can be overridden)
    "dt": 0.1,                          # Time step in seconds
    "solver_method": "RK45",            # ODE solver method ('RK45' is fastest, 'LSODA' for stiff problems)
    "rtol": ACCURACY_PRESETS[DEFAULT_ACCURACY]["rtol"],  # Relative tolerance for solver
    "atol": ACCURACY_PRESETS[DEFAULT_ACCURACY]["atol"],  # Absolute tolerance for solver
}

# Generate time span array
SIMULATION_SETTINGS["T_span"] = np.arange(
    SIMULATION_SETTINGS["t_start"],
    SIMULATION_SETTINGS["t_end"] + SIMULATION_SETTINGS["dt"],
    SIMULATION_SETTINGS["dt"]
)

# =============================================================================
# CONTROL GAINS (PD Controller)
# =============================================================================
CONTROL_GAINS = {
    "kp": 2.0,      # Proportional gain
    "kd": 1.5,      # Derivative gain
    "ki": 0.0       # Integral gain (not used currently)
}


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
INITIAL_CONDITIONS = {
    "X_0": np.array([0.0]),        # Global X coordinate origin
    "Y_0": np.array([0.0]),        # Global Y coordinate origin
    "starting_point": np.array([0, 0]),  # Starting point of link 1
    "theta_n_0": np.deg2rad(0.0),  # Initial head angle in radians
    "phi_1_0": np.deg2rad(0.0),    # Initial first joint angle (default: 0 degrees)
}


# =============================================================================
# STEERING OFFSETS (Time-dependent joint offsets for turning)
# =============================================================================
STEERING_CONFIG = {
    "turn_1": {
        "t_start": 20.0,
        "t_end": 30.0,
        "offset": np.deg2rad(5)     # Turn right
    },
    "turn_2": {
        "t_start": 50.0,
        "t_end": 60.0,
        "offset": np.deg2rad(-10)   # Turn left
    }
}


# =============================================================================
# PLOTTING SETTINGS
# =============================================================================
PLOT_SETTINGS = {
    "figure_dpi": 400,
    "save_figures": True,
    "show_figures": True,
    "font_size": 14,
    "line_width": 1.5,
    "colormap": "tab10"
}


def get_snake_parameters():
    """
    Combine motion parameters with time span for compatibility with existing code.
    
    Returns:
        dict: Complete snake parameters dictionary
    """
    params = SNAKE_MOTION_PARAMS.copy()
    params["T_span"] = SIMULATION_SETTINGS["T_span"]
    return params


def get_all_config():
    """
    Get all configuration as a single dictionary.
    
    Returns:
        dict: All configuration parameters
    """
    return {
        "physical": PHYSICAL_PROPERTIES,
        "motion": SNAKE_MOTION_PARAMS,
        "simulation": SIMULATION_SETTINGS,
        "control": CONTROL_GAINS,
        "initial": INITIAL_CONDITIONS,
        "steering": STEERING_CONFIG,
        "plotting": PLOT_SETTINGS
    }


def update_config(num_links=None, simulation_time=None, accuracy=None, initial_joint_angle=None):
    """
    Update configuration with user-specified values.
    
    Args:
        num_links: Number of links in the snake robot (default: 10)
        simulation_time: Total simulation time in seconds (default: 60.0)
        accuracy: Accuracy level ('low', 'medium', 'high')
        initial_joint_angle: Initial angle for first joint in degrees (default: 0.0)
    
    Returns:
        tuple: Updated (PHYSICAL_PROPERTIES, SIMULATION_SETTINGS)
    """
    global PHYSICAL_PROPERTIES, SIMULATION_SETTINGS, INITIAL_CONDITIONS
    
    if num_links is not None:
        PHYSICAL_PROPERTIES["N"] = int(num_links)
    
    if simulation_time is not None:
        SIMULATION_SETTINGS["t_end"] = float(simulation_time)
        # Regenerate time span array
        SIMULATION_SETTINGS["T_span"] = np.arange(
            SIMULATION_SETTINGS["t_start"],
            SIMULATION_SETTINGS["t_end"] + SIMULATION_SETTINGS["dt"],
            SIMULATION_SETTINGS["dt"]
        )
    
    if accuracy is not None and accuracy in ACCURACY_PRESETS:
        SIMULATION_SETTINGS["rtol"] = ACCURACY_PRESETS[accuracy]["rtol"]
        SIMULATION_SETTINGS["atol"] = ACCURACY_PRESETS[accuracy]["atol"]
    
    if initial_joint_angle is not None:
        INITIAL_CONDITIONS["phi_1_0"] = np.deg2rad(float(initial_joint_angle))
    
    return PHYSICAL_PROPERTIES, SIMULATION_SETTINGS


def get_user_input():
    """
    Prompt user for number of links, simulation time, accuracy level, and initial joint angle.
    
    Returns:
        tuple: (num_links, simulation_time, accuracy, initial_joint_angle)
    """
    print("\n" + "=" * 60)
    print("SNAKE ROBOT SIMULATION - CONFIGURATION")
    print("=" * 60)
    
    # Get number of links
    while True:
        try:
            num_links_input = input(f"\nEnter number of links [{DEFAULT_NUM_LINKS}]: ").strip()
            if num_links_input == "":
                num_links = DEFAULT_NUM_LINKS
            else:
                num_links = int(num_links_input)
            
            if num_links < 2:
                print("Error: Number of links must be at least 2.")
                continue
            if num_links > 50:
                print("Warning: Large number of links may result in slow simulation.")
            break
        except ValueError:
            print("Error: Please enter a valid integer.")
    
    # Get simulation time
    while True:
        try:
            sim_time_input = input(f"Enter simulation time in seconds [{DEFAULT_SIMULATION_TIME}]: ").strip()
            if sim_time_input == "":
                simulation_time = DEFAULT_SIMULATION_TIME
            else:
                simulation_time = float(sim_time_input)
            
            if simulation_time <= 0:
                print("Error: Simulation time must be positive.")
                continue
            if simulation_time > 300:
                print("Warning: Long simulation time may take a while to compute.")
            break
        except ValueError:
            print("Error: Please enter a valid number.")
    
    # Get accuracy level
    print("\nAccuracy levels:")
    for key, preset in ACCURACY_PRESETS.items():
        marker = " <-- default" if key == DEFAULT_ACCURACY else ""
        print(f"  [{key}] {preset['name'].capitalize()}: {preset['description']}{marker}")
    
    while True:
        accuracy_input = input(f"Select accuracy level (1/2/3) [{DEFAULT_ACCURACY}]: ").strip()
        if accuracy_input == "":
            accuracy = DEFAULT_ACCURACY
            break
        elif accuracy_input in ACCURACY_PRESETS:
            accuracy = accuracy_input
            break
        else:
            print("Error: Please enter 1, 2, or 3.")
    
    # Get initial joint angle
    default_angle = 0.0
    while True:
        try:
            angle_input = input(f"Enter initial joint angle phi_1 in degrees [{default_angle}]: ").strip()
            if angle_input == "":
                initial_joint_angle = default_angle
            else:
                initial_joint_angle = float(angle_input)
            
            if abs(initial_joint_angle) > 90:
                print("Warning: Large initial angle may cause instability.")
            break
        except ValueError:
            print("Error: Please enter a valid number.")
    
    print("\n" + "-" * 60)
    print("Configuration Summary:")
    print(f"  Number of links: {num_links}")
    print(f"  Simulation time: {simulation_time} seconds")
    print(f"  Accuracy: {ACCURACY_PRESETS[accuracy]['name']} (rtol={ACCURACY_PRESETS[accuracy]['rtol']}, atol={ACCURACY_PRESETS[accuracy]['atol']})")
    print(f"  Initial joint angle (phi_1): {initial_joint_angle} degrees")
    print("=" * 60)
    
    return num_links, simulation_time, accuracy, initial_joint_angle


if __name__ == "__main__":
    # Print configuration when run directly
    print("Snake Robot Configuration")
    print("=" * 50)
    print(f"\nPhysical Properties: {PHYSICAL_PROPERTIES}")
    print(f"\nMotion Parameters: {SNAKE_MOTION_PARAMS}")
    print(f"\nSimulation Settings: {SIMULATION_SETTINGS}")
    print(f"\nControl Gains: {CONTROL_GAINS}")
