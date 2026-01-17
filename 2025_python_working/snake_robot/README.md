# Snake Robot Simulation

A modular Python implementation of snake robot dynamics, control, and simulation.

## Project Structure

```
snake_robot/
├── __init__.py          # Package initialization
├── config.py            # Configuration parameters
├── snake_init.py        # Robot initialization
├── dynamic_model.py     # Dynamics and ODE solver
├── plotting.py          # Visualization functions
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the full simulation:
```bash
python main.py
```

### Programmatic Usage

```python
from snake_robot import SnakeInitializer, SnakeDynamicModel, plotting

# Initialize
snake = SnakeInitializer()
params = snake.get_parameters()

# Create model and simulate
model = SnakeDynamicModel(*params)
ref_angles, _, _, T_ref = model.generate_reference_angles()
T, q = model.simulate()

# Extract and plot results
states = model.extract_states(q)
plotting.plot_cm_trajectory(states['px'], states['py'])
```

### Customizing Parameters

Edit `config.py` to change:

- **Physical properties**: Link length, mass, friction coefficients
- **Motion parameters**: Amplitude, frequency, phase offset
- **Control gains**: PD controller gains
- **Simulation settings**: Time span, solver tolerances
- **Steering configuration**: Time-dependent turning offsets

Example:
```python
# In config.py
PHYSICAL_PROPERTIES = {
    "l": 0.07,          # Half-link length (m)
    "m": 1.0,           # Mass per link (kg)
    "N": 10,            # Number of links
    "c_n": 10.0,        # Normal friction
    "c_t": 1.0,         # Tangential friction
}

CONTROL_GAINS = {
    "kp": 2.0,          # Proportional gain
    "kd": 1.5,          # Derivative gain
}
```

## Output Files

The simulation generates the following plots:

| File | Description |
|------|-------------|
| `Reference_vs_Actual_Joints.png` | Joint angle tracking comparison |
| `Snake_CM_Trajectory.png` | Center of mass x-y trajectory |
| `Position_vs_Time.png` | CM position components vs time |
| `Head_Angle.png` | Head link orientation |
| `Tracking_Error.png` | Joint angle tracking errors |
| `Simulation_Summary.png` | Combined summary figure |

## Mathematical Model

The snake robot is modeled using:

- **Kinematics**: Link positions computed from joint angles using coordinate transformations
- **Dynamics**: Lagrangian mechanics with anisotropic friction
- **Control**: PD controller for joint angle tracking with feedforward

State vector: `q = [φ₁, ..., φₙ₋₁, θₙ, pₓ, pᵧ, φ̇₁, ..., φ̇ₙ₋₁, θ̇ₙ, ṗₓ, ṗᵧ]`

where:
- `φᵢ` - Joint angles (actuated)
- `θₙ` - Head link absolute angle (unactuated)
- `(pₓ, pᵧ)` - Center of mass position (unactuated)

## Reference

Based on the snake robotics theory from:
- "Snake Robotics" textbook
- Numerical methods from SciPy documentation

## License

MIT License
