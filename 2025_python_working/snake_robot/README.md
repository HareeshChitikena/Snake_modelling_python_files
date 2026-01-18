# 🐍 Snake Robot Dynamic Simulation

A high-performance Python implementation of snake robot dynamics, control, and simulation with real-time visualization. This project provides an accurate dynamic model based on the snake robotics textbook, optimized to run **400-500x faster than real-time**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

### 🚀 Performance
- **400-500x Real-Time Factor (RTF)**: 80-second simulation completes in ~0.17 seconds
- **Optimized ODE Solver**: RK45 with configurable accuracy presets
- **Efficient Matrix Operations**: Vectorized NumPy computations

### 📊 Visualization
- **8 Comprehensive Plot Types**: Joint angles, trajectories, tracking errors, and more
- **Steering Event Markers**: Red triangles mark turning points on CM trajectory
- **Color-Coded Time Progression**: Viridis colormap shows simulation progress
- **Average Joint Angle Plot**: Visualize overall body curvature (φ̄)

### ⚙️ Configuration
- **User-Friendly Input**: Interactive prompts for key parameters
- **Flexible Accuracy Presets**: Low/Medium/High accuracy options
- **Configurable Steering**: Define custom turning maneuvers

---

## 📁 Project Structure

```
snake_robot/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration parameters & user input
├── snake_init.py        # Robot initialization (geometry, state)
├── dynamic_model.py     # Core dynamics & ODE solver
├── plotting.py          # All visualization functions (9 plot types)
├── main.py              # Main entry point with CLI
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
│
└── output/              # Generated outputs (auto-created)
    ├── plots/           # Static plot images
    │   ├── Reference_vs_Actual_Joints.png
    │   ├── Snake_CM_Trajectory.png
    │   ├── Position_vs_Time.png
    │   ├── Head_Angle.png
    │   ├── Average_Joint_Angle.png
    │   ├── Tracking_Error.png
    │   ├── Simulation_Summary.png
    │   ├── Performance_Metrics.png
    │   └── Snake_Motion_Sequence.png
    │
    └── animations/      # Animation files
        └── snake_robot_animation.gif
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/snake-robot-simulation.git
cd snake-robot-simulation/snake_robot

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
tqdm>=4.62.0
```

---

## 🚀 Quick Start

### Interactive Mode (Recommended)

```bash
cd snake_robot
python main.py
```

You'll be prompted for:
1. **Number of links** (default: 10)
2. **Simulation time** in seconds (default: 60)
3. **Accuracy level**: 1=Low (fast), 2=Medium, 3=High
4. **Initial joint angle** in degrees (default: 0°)

### Command Line (Non-Interactive)

```bash
# Run with default settings
python -c "from main import run_simulation, create_plots; results = run_simulation(verbose=True); create_plots(results)"

# Custom parameters: 10 links, 80 seconds, low accuracy, 0° initial angle
python -c "from main import run_simulation, create_plots; results = run_simulation(num_links=10, simulation_time=80, accuracy='1', initial_joint_angle=0, verbose=True); create_plots(results, show=True, save=True)"

# Quick 5-second test
python -c "from main import run_simulation, create_plots; results = run_simulation(simulation_time=5, accuracy='1', verbose=True); create_plots(results)"
```

---

## 📖 API Reference

### Main Functions

#### `run_simulation()`
```python
from main import run_simulation

results = run_simulation(
    num_links=10,           # Number of robot links (default: 10)
    simulation_time=60,     # Simulation duration in seconds
    accuracy='1',           # '1'=Low, '2'=Medium, '3'=High
    initial_joint_angle=0,  # Initial φ₁ in degrees
    verbose=True            # Print progress messages
)
```

**Returns:** Dictionary containing:
- `time`: Time array
- `states`: Dict with `phi`, `px`, `py`, `theta_n`, etc.
- `ref_angles`: Reference joint angles array
- `T_ref`: Reference time array
- `performance`: Performance metrics dict

#### `create_plots()`
```python
from main import create_plots

create_plots(
    results,      # Results dict from run_simulation()
    show=True,    # Display plots interactively
    save=True     # Save plots as PNG files
)
```

### Individual Plot Functions

```python
from snake_robot import plotting

# All functions accept save=True/False and show=True/False

plotting.plot_reference_vs_actual(ref_angles, T_ref, actual_angles, T_actual)
plotting.plot_cm_trajectory(px, py, time=T)
plotting.plot_position_vs_time(time, px, py)
plotting.plot_head_angle(time, theta_n)
plotting.plot_average_joint_angle(time, joint_angles, ref_angles, T_ref)  # NEW
plotting.plot_tracking_error(time, ref_angles, actual_angles)
plotting.create_summary_plot(time, states, ref_angles, T_ref)
plotting.plot_performance_metrics(performance_dict)
```

---

## 📊 Output Plots

| Plot | Description | Filename |
|------|-------------|----------|
| **Reference vs Actual** | Compares reference and simulated joint angles (degrees) | `Reference_vs_Actual_Joints.png` |
| **CM Trajectory** | X-Y path with steering markers (🔺) and time colormap | `Snake_CM_Trajectory.png` |
| **Position vs Time** | X and Y position over time | `Position_vs_Time.png` |
| **Head Angle** | Orientation of the head link (θₙ) | `Head_Angle.png` |
| **Average Joint Angle** | Mean joint angle φ̄ with steering regions | `Average_Joint_Angle.png` |
| **Tracking Error** | Difference between reference and actual angles | `Tracking_Error.png` |
| **Summary** | 4-panel overview of key metrics | `Simulation_Summary.png` |
| **Performance** | Bar chart of simulation vs wall-clock time | `Performance_Metrics.png` |

---

## ⚙️ Configuration

### Steering Configuration (`config.py`)

```python
STEERING_CONFIG = {
    "turn_1": {
        "t_start": 20,      # Start time (seconds)
        "t_end": 30,        # End time (seconds)
        "offset": np.deg2rad(5)   # +5° = turn right
    },
    "turn_2": {
        "t_start": 50,
        "t_end": 60,
        "offset": np.deg2rad(-10)  # -10° = turn left
    }
}
```

### Accuracy Presets

| Level | rtol | atol | Speed | Use Case |
|-------|------|------|-------|----------|
| `'1'` (Low) | 1e-2 | 1e-2 | ⚡ Fastest | Quick tests, debugging |
| `'2'` (Medium) | 1e-4 | 1e-4 | ⚖️ Balanced | General use |
| `'3'` (High) | 1e-6 | 1e-6 | 🎯 Accurate | Final results, validation |

### Physical Parameters

```python
PHYSICAL_PROPERTIES = {
    'N': 10,              # Number of links
    'l': 0.14,            # Link length (m)
    'm': 1.0,             # Link mass (kg)
    'J': 0.0016,          # Moment of inertia (kg·m²)
    'ct': 1.0,            # Tangential friction
    'cn': 10.0,           # Normal friction (cn >> ct for anisotropic)
}
```

---

## 🔧 Key Improvements (2025 Version)

### Bug Fixes
1. **X velocity sign correction**: Fixed `X_d = +l*K'*S_theta*theta_d` (was negative)
2. **Y velocity sign correction**: Fixed `Y_d = -l*K'*C_theta*theta_d` (was positive)
3. **W_term calculation**: Fixed `theta_d²` term (was using wrong variable)

### Performance Enhancements
- Optimized from ~1x RTF to **400-500x RTF**
- 80-second simulation in 0.17 seconds wall-clock time

### New Features
- **Interactive user input** for simulation parameters
- **Initial joint angle** configuration (degrees)
- **Steering event markers** on CM trajectory plot
- **Average joint angle plot** (φ̄) with steering region highlighting
- **Performance metrics** display and visualization
- **Color-coded trajectory** showing time progression

### Plot Improvements
- Joint angles displayed in **degrees** (not radians)
- **Removed confusing arrows** from CM trajectory
- Added **Viridis colormap** for time progression
- **Steering markers**: 🔺 at turn start, 🔻 at turn end

---

## 📈 Performance Benchmarks

| Simulation Time | Wall-Clock Time | RTF | Time Steps |
|-----------------|-----------------|-----|------------|
| 5 s | 0.012 s | 417x | 51 |
| 60 s | 0.145 s | 414x | 601 |
| 80 s | 0.170 s | 470x | 801 |

*Tested on: Intel Core i7, 16GB RAM, Windows 11, Python 3.13*

---

## 📚 Theory & References

This simulation implements the snake robot dynamics from:

> **Snake Robotics** - Pål Liljebäck, Kristin Y. Pettersen, Øyvind Stavdahl, Jan Tommy Gravdahl

### Key Equations

**Joint angle reference (serpenoid curve):**

$$\phi_i^{ref}(t) = \alpha \sin(\omega t + (i-1)\delta) + \phi_0$$

**Average joint angle:**

$$\bar{\phi}(t) = \frac{1}{n} \sum_{i=1}^{n} \phi_i(t)$$

**Dynamics:**
The model solves the full nonlinear ODE system including:
- Anisotropic ground friction (ct, cn)
- Link inertia and coupling
- PD joint controllers

---

## 🐛 Troubleshooting

### Common Issues

**Import Error:**
```bash
# Make sure you're in the snake_robot directory
cd snake_robot
python main.py
```

**Slow Simulation:**
```python
# Use low accuracy for faster results
results = run_simulation(accuracy='1')
```

**Plot Not Showing:**
```python
# Ensure matplotlib backend is configured
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Last updated: January 2026*
