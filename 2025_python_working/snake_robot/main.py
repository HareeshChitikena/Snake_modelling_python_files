#!/usr/bin/env python
"""
Snake Robot Simulation - Main Entry Point
==========================================

This script runs the full snake robot simulation including:
1. Initialization of robot parameters and state
2. Generation of reference trajectories
3. Dynamic simulation with ODE solver
4. Visualization of results

Usage:
    python main.py
    
    The program will prompt for:
    - Number of links (default: 10)
    - Simulation time in seconds (default: 60)

For other configurations, modify the snake_robot/config.py file.

Author: Snake Robot Project
Reference: Snake Robotics textbook
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from snake_robot import (
        SnakeInitializer,
        SnakeDynamicModel,
        plotting,
        PHYSICAL_PROPERTIES,
        SIMULATION_SETTINGS
    )
    from snake_robot.config import get_user_input, update_config
except ImportError:
    # When running from within the snake_robot directory
    from snake_init import SnakeInitializer
    from dynamic_model import SnakeDynamicModel
    import plotting
    from config import PHYSICAL_PROPERTIES, SIMULATION_SETTINGS, get_user_input, update_config


def run_simulation(num_links=None, simulation_time=None, accuracy=None, initial_joint_angle=None, verbose=True):
    """
    Run the complete snake robot simulation.
    
    Args:
        num_links: Number of links (None to use config default)
        simulation_time: Simulation time in seconds (None to use config default)
        accuracy: Accuracy level ('low', 'medium', 'high') (None to use config default)
        initial_joint_angle: Initial angle for first joint in degrees (None to use default: 0)
        verbose: Whether to print progress messages
    Returns:
        dict: Simulation results including time, states, and references
    """
    # Update configuration with user-specified values
    if num_links is not None or simulation_time is not None or accuracy is not None or initial_joint_angle is not None:
        update_config(num_links=num_links, simulation_time=simulation_time, 
                     accuracy=accuracy, initial_joint_angle=initial_joint_angle)
    
    if verbose:
        print("=" * 60)
        print("SNAKE ROBOT SIMULATION")
        print("=" * 60)
    
    # =========================================================================
    # Step 1: Initialize snake robot
    # =========================================================================
    if verbose:
        print("\n[1/4] Initializing snake robot...")
    
    snake = SnakeInitializer()
    phy_props, init_vals, snake_params, control_gains = snake.get_parameters()
    
    if verbose:
        snake.print_summary()
    
    # =========================================================================
    # Step 2: Create dynamic model
    # =========================================================================
    if verbose:
        print("\n[2/4] Creating dynamic model...")
    
    model = SnakeDynamicModel(phy_props, init_vals, snake_params, control_gains)
    
    # =========================================================================
    # Step 3: Generate reference angles
    # =========================================================================
    if verbose:
        print("\n[3/4] Generating reference angles...")
    
    ref_angles, ref_angles_d, ref_angles_dd, T_ref = model.generate_reference_angles()
    
    # Convert dict to array for plotting
    n_joints = phy_props['N'] - 1
    ref_array = np.array([ref_angles[f'phi{i+1}'] for i in range(n_joints)])
    
    # =========================================================================
    # Step 4: Run simulation
    # =========================================================================
    if verbose:
        print("\n[4/4] Running dynamic simulation...")
    
    import time as time_module
    start_wall_time = time_module.time()
    
    T, q = model.simulate(verbose=verbose)
    
    end_wall_time = time_module.time()
    wall_clock_time = end_wall_time - start_wall_time
    
    # Extract states
    states = model.extract_states(q)
    
    # Calculate performance metrics
    simulation_time = T[-1] - T[0]
    real_time_factor = simulation_time / wall_clock_time if wall_clock_time > 0 else float('inf')
    time_per_step = wall_clock_time / len(T) * 1000  # in milliseconds
    
    performance = {
        'simulation_time': simulation_time,
        'wall_clock_time': wall_clock_time,
        'real_time_factor': real_time_factor,
        'time_steps': len(T),
        'time_per_step_ms': time_per_step,
        'num_links': phy_props['N'],
        'function_evaluations': getattr(model, '_last_nfev', 'N/A')
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Time range: {T[0]:.1f} to {T[-1]:.1f} seconds")
        print(f"  Time steps: {len(T)}")
        print(f"  Final CM position: ({states['px'][-1]:.3f}, {states['py'][-1]:.3f}) m")
        print(f"  Total X displacement: {states['px'][-1] - states['px'][0]:.3f} m")
        print(f"  Total Y displacement: {states['py'][-1] - states['py'][0]:.3f} m")
        
        print("\n" + "-" * 60)
        print("PERFORMANCE METRICS")
        print("-" * 60)
        print(f"  Simulation time (requested): {simulation_time:.2f} seconds")
        print(f"  Wall-clock time (actual):    {wall_clock_time:.4f} seconds")
        print(f"  Real-time factor (RTF):      {real_time_factor:.1f}x")
        print(f"  Time per step:               {time_per_step:.3f} ms")
        print("-" * 60)
        if real_time_factor > 1:
            print(f"  ✓ Simulation ran {real_time_factor:.0f}x FASTER than real-time!")
        else:
            print(f"  ✗ Simulation ran {1/real_time_factor:.1f}x SLOWER than real-time")
    
    return {
        'time': T,
        'states': states,
        'ref_angles': ref_array,
        'T_ref': T_ref,
        'model': model,
        'phy_props': phy_props,
        'performance': performance
    }


def create_plots(results, show=True, save=True):
    """
    Create all visualization plots from simulation results.
    
    Args:
        results: Dictionary from run_simulation()
        show: Whether to display plots
        save: Whether to save plots to files
    """
    T = results['time']
    states = results['states']
    ref_array = results['ref_angles']
    T_ref = results['T_ref']
    
    print("\nGenerating plots...")
    
    # 1. Reference vs Actual joint angles
    print("  - Reference vs Actual angles...")
    plotting.plot_reference_vs_actual(
        ref_array, T_ref, 
        states['phi'], T,
        save=save, show=show
    )
    
    # 2. Center of mass trajectory
    print("  - CM Trajectory...")
    plotting.plot_cm_trajectory(
        states['px'], states['py'], 
        time=T,
        save=save, show=show
    )
    
    # 3. Position vs time
    print("  - Position vs Time...")
    plotting.plot_position_vs_time(
        T, states['px'], states['py'],
        save=save, show=show
    )
    
    # 4. Head angle
    print("  - Head angle...")
    plotting.plot_head_angle(
        T, states['theta_n'],
        save=save, show=show
    )
    
    # 5. Average joint angle (phi_bar)
    print("  - Average joint angle (phi_bar)...")
    plotting.plot_average_joint_angle(
        T, states['phi'], 
        ref_angles=ref_array, T_ref=T_ref,
        save=save, show=show
    )
    
    # 6. Tracking error
    print("  - Tracking error...")
    plotting.plot_tracking_error(
        T, ref_array, states['phi'],
        save=save, show=show
    )
    
    # 7. Summary plot
    print("  - Summary plot...")
    plotting.create_summary_plot(
        T, states, ref_array, T_ref,
        save=save, show=show
    )
    
    # 8. Performance metrics (if available)
    if 'performance' in results:
        print("  - Performance metrics...")
        plotting.plot_performance_metrics(
            results['performance'],
            save=save, show=show
        )
    
    print("\nAll plots generated!")


def create_animation(results, save=True, show=True, speed_factor=1.0):
    """
    Create animation of the snake robot movement.
    
    Args:
        results: Dictionary from run_simulation()
        save: Whether to save animation as GIF
        show: Whether to display animation
        speed_factor: Playback speed multiplier
    """
    T = results['time']
    states = results['states']
    phy_props = results['phy_props']
    
    print("\nGenerating animation...")
    print("  This may take a moment for long simulations...")
    
    # Create the animation
    anim = plotting.animate_snake_robot(
        T, states, phy_props,
        save=save, show=show,
        speed_factor=speed_factor
    )
    
    return anim


def create_motion_sequence(results, n_frames=6, save=True, show=True):
    """
    Create a multi-panel figure showing snake at different time points.
    
    Useful for publications when animation is not possible.
    
    Args:
        results: Dictionary from run_simulation()
        n_frames: Number of frames to show
        save: Whether to save figure
        show: Whether to display figure
    """
    T = results['time']
    states = results['states']
    phy_props = results['phy_props']
    
    print(f"\nGenerating motion sequence with {n_frames} frames...")
    
    fig, axes = plotting.create_snake_frames(
        T, states, phy_props,
        n_frames=n_frames,
        save=save, show=show
    )
    
    return fig, axes


def main(interactive=True):
    """
    Main function to run the complete simulation and visualization.
    
    Args:
        interactive: If True, prompt user for inputs. If False, use defaults.
    """
    
    # Get user input for configuration
    if interactive:
        num_links, simulation_time, accuracy, initial_joint_angle = get_user_input()
    else:
        num_links = None
        simulation_time = None
        accuracy = None
        initial_joint_angle = None
    
    # Run simulation
    results = run_simulation(
        num_links=num_links, 
        simulation_time=simulation_time,
        accuracy=accuracy,
        initial_joint_angle=initial_joint_angle,
        verbose=True
    )
    
    # Create plots
    create_plots(results, show=True, save=True)
    
    # Create motion sequence (static frames)
    create_motion_sequence(results, n_frames=6, save=True, show=True)
    
    # Create animation (GIF)
    create_animation(results, save=True, show=False)
    
    print("\n" + "=" * 60)
    print("Output files saved:")
    print("  - Reference_vs_Actual_Joints.png")
    print("  - Snake_CM_Trajectory.png")
    print("  - Position_vs_Time.png")
    print("  - Head_Angle.png")
    print("  - Average_Joint_Angle.png")
    print("  - Tracking_Error.png")
    print("  - Simulation_Summary.png")
    print("  - Performance_Metrics.png")
    print("  - Snake_Motion_Sequence.png")
    print("  - snake_robot_animation.gif")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main(interactive=True)
