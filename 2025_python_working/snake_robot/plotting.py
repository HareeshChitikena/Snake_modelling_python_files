"""
Snake Robot Plotting Module
===========================
Provides visualization functions for snake robot simulation results:
- Joint angle plots (reference vs actual)
- Center of mass trajectory
- Animation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import colormaps

try:
    from .config import PLOT_SETTINGS
except ImportError:
    from config import PLOT_SETTINGS


def setup_plot_style():
    """Configure matplotlib plotting style."""
    plt.rcParams['font.size'] = PLOT_SETTINGS.get('font_size', 14)
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['figure.dpi'] = 100


def plot_reference_angles(time, angles_dict, n_links=9, save=True, show=True):
    """
    Plot reference joint angles over time.
    
    Args:
        time: Time array
        angles_dict: Dictionary with joint angles {'phi1': array, 'phi2': array, ...}
        n_links: Number of joints to plot
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = colormaps['tab10']
    
    for i in range(1, n_links + 1):
        key = f'phi{i}'
        if key in angles_dict:
            ax.plot(time, angles_dict[key], linewidth=1.5, color=colors(i-1),
                   label=rf'$\phi_{{{i}}}$')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Joint Angle (rad)', fontsize=14)
    ax.set_title('Reference Joint Angles', fontsize=16)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-0.8, 0.8)
    ax.legend(loc='upper right', fontsize=10, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Reference_Angles.png', dpi=PLOT_SETTINGS['figure_dpi'], 
                   bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def plot_reference_vs_actual(ref_angles, T_ref, actual_angles, T_actual, 
                             n_links=9, save=True, show=True):
    """
    Plot reference vs actual joint angles comparison.
    
    Args:
        ref_angles: Reference angles array (n_links x time_steps) in RADIANS
        T_ref: Reference time array
        actual_angles: Actual angles array from simulation in RADIANS
        T_actual: Simulation time array
        n_links: Number of joints to plot
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    colors = colormaps['tab10']
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert radians to degrees for plotting
    ref_deg = np.rad2deg(ref_angles)
    actual_deg = np.rad2deg(actual_angles)
    
    for i in range(min(n_links, ref_angles.shape[0])):
        # Reference (solid line)
        ax.plot(T_ref, ref_deg[i, :], linewidth=1.5, color=colors(i),
               label=rf'$\phi_{{{i+1}}}$ Reference')
        
        # Actual (dashed line)
        if i < actual_angles.shape[0]:
            ax.plot(T_actual, actual_deg[i, :], '--', linewidth=1.5, 
                   color=colors(i), label=rf'$\phi_{{{i+1}}}$ Actual')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, T_ref[-1]//10)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(max(0.5, T_ref[-1]//20)))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Joint Angle (degrees)', fontsize=14)
    ax.set_title('Reference vs Actual Joint Angles', fontsize=16)
    ax.set_xlim(T_actual[0], T_ref[-1])
    ax.set_ylim(-50, 50)
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Reference_vs_Actual_Joints.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def plot_cm_trajectory(px, py, time=None, save=True, show=True):
    """
    Plot the x-y trajectory of the snake's center of mass.
    
    Args:
        px: X position array over time
        py: Y position array over time
        time: Optional time array for time markers
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    # Import steering config to mark steering events
    try:
        from .config import STEERING_CONFIG
    except ImportError:
        from config import STEERING_CONFIG
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectory with color gradient to show time progression
    # Create segments for color mapping
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use a colormap to show time progression
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(0, len(px)-1)
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5)
    lc.set_array(np.arange(len(px)))
    line = ax.add_collection(lc)
    
    # Add colorbar to show time
    cbar = fig.colorbar(line, ax=ax, shrink=0.8)
    if time is not None:
        cbar.set_label('Time (s)', fontsize=12)
        # Set colorbar ticks to show actual time values
        n_ticks = 5
        tick_indices = np.linspace(0, len(px)-1, n_ticks).astype(int)
        cbar.set_ticks(tick_indices)
        cbar.set_ticklabels([f'{time[i]:.1f}' for i in tick_indices])
    else:
        cbar.set_label('Time step', fontsize=12)
    
    # Mark start and end points
    ax.plot(px[0], py[0], 'go', markersize=14, label='Start', zorder=5, 
            markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(px[-1], py[-1], 'bs', markersize=14, label='End', zorder=5,
            markeredgecolor='darkblue', markeredgewidth=2)
    
    # Mark steering events with red triangles
    if time is not None:
        steering_marked = False
        for turn_name, turn_config in STEERING_CONFIG.items():
            t_start = turn_config["t_start"]
            t_end = turn_config["t_end"]
            offset_deg = np.rad2deg(turn_config["offset"])
            
            # Check if steering occurs within simulation time
            if t_start <= time[-1]:
                # Mark start of steering
                idx_start = np.argmin(np.abs(time - t_start))
                if idx_start < len(px):
                    label = 'Steering' if not steering_marked else None
                    ax.plot(px[idx_start], py[idx_start], 'r^', markersize=12, 
                           label=label, zorder=6, markeredgecolor='darkred', markeredgewidth=1.5)
                    
                    # Add annotation with time and offset
                    direction = "Right" if offset_deg > 0 else "Left"
                    ax.annotate(f't={t_start:.0f}s\n{direction} ({offset_deg:+.0f}°)', 
                               (px[idx_start], py[idx_start]),
                               textcoords="offset points", xytext=(10, 10),
                               fontsize=9, fontweight='bold', color='red',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor='red', alpha=0.8))
                    steering_marked = True
                
                # Mark end of steering (if within time range)
                if t_end <= time[-1]:
                    idx_end = np.argmin(np.abs(time - t_end))
                    if idx_end < len(px):
                        ax.plot(px[idx_end], py[idx_end], 'rv', markersize=10, 
                               zorder=6, markeredgecolor='darkred', markeredgewidth=1.5)
                        ax.annotate(f't={t_end:.0f}s\n(end)', 
                                   (px[idx_end], py[idx_end]),
                                   textcoords="offset points", xytext=(10, -15),
                                   fontsize=8, color='darkred', alpha=0.8)
    
    # Add time markers (small dots at regular intervals, but not where steering is marked)
    if time is not None and len(time) > 10:
        t_max = time[-1]
        marker_interval = max(1, int(t_max / 5))
        time_markers = np.arange(marker_interval, t_max, marker_interval)
        
        # Get steering times to avoid overlap
        steering_times = []
        for turn_config in STEERING_CONFIG.values():
            steering_times.extend([turn_config["t_start"], turn_config["t_end"]])
        
        for t_mark in time_markers:
            # Skip if too close to a steering marker
            if any(abs(t_mark - st) < 2 for st in steering_times):
                continue
            idx = np.argmin(np.abs(time - t_mark))
            if idx < len(px) and idx > 0:
                ax.plot(px[idx], py[idx], 'ko', markersize=5, zorder=4, alpha=0.6)
                ax.annotate(f'{t_mark:.0f}s', (px[idx], py[idx]), 
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=0.7)
    
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.set_title('Snake Robot Center of Mass Trajectory', fontsize=16)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Auto-scale with some padding
    x_margin = (max(px) - min(px)) * 0.1 + 0.05
    y_margin = (max(py) - min(py)) * 0.1 + 0.05
    ax.set_xlim(min(px) - x_margin, max(px) + x_margin)
    ax.set_ylim(min(py) - y_margin, max(py) + y_margin)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Snake_CM_Trajectory.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def plot_position_vs_time(time, px, py, save=True, show=True):
    """
    Plot X and Y positions of center of mass vs time.
    
    Args:
        time: Time array
        px: X position array
        py: Y position array
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # X position
    axes[0].plot(time, px, 'b-', linewidth=1.5, label='X position')
    axes[0].set_ylabel('X (m)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Center of Mass Position vs Time', fontsize=14)
    
    # Y position
    axes[1].plot(time, py, 'r-', linewidth=1.5, label='Y position')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Y (m)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Position_vs_Time.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, axes


def plot_velocity_vs_time(time, px_d, py_d, save=True, show=True):
    """
    Plot velocity components vs time.
    
    Args:
        time: Time array
        px_d: X velocity array
        py_d: Y velocity array
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # X velocity
    axes[0].plot(time, px_d, 'b-', linewidth=1.5, label=r'$\dot{p}_x$')
    axes[0].set_ylabel(r'$\dot{p}_x$ (m/s)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Center of Mass Velocity vs Time', fontsize=14)
    
    # Y velocity  
    axes[1].plot(time, py_d, 'r-', linewidth=1.5, label=r'$\dot{p}_y$')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel(r'$\dot{p}_y$ (m/s)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Velocity_vs_Time.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, axes


def plot_head_angle(time, theta_n, save=True, show=True):
    """
    Plot head link angle vs time.
    
    Args:
        time: Time array
        theta_n: Head angle array
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(time, np.rad2deg(theta_n), 'g-', linewidth=1.5, label=r'$\theta_N$ (head)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Head Angle (deg)', fontsize=12)
    ax.set_title('Head Link Orientation', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Head_Angle.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def plot_average_joint_angle(time, joint_angles, ref_angles=None, T_ref=None, 
                             save=True, show=True):
    """
    Plot the average joint angle (φ̄ or phi_bar) over time.
    
    The average joint angle is computed as the mean of all joint angles at each
    time step: φ̄(t) = (1/n) * Σ φ_i(t)
    
    This is an important metric for snake robot locomotion as it indicates
    the overall body curvature/orientation tendency.
    
    Args:
        time: Time array
        joint_angles: Joint angles array (n_joints x time_steps) in RADIANS
        ref_angles: Optional reference angles array for comparison
        T_ref: Optional reference time array
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    setup_plot_style()
    
    # Import steering config to mark steering events
    try:
        from .config import STEERING_CONFIG
    except ImportError:
        from config import STEERING_CONFIG
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Compute average joint angle (mean across all joints at each time step)
    # joint_angles has shape (n_joints, time_steps)
    phi_avg = np.mean(joint_angles, axis=0)  # Average across joints
    phi_avg_deg = np.rad2deg(phi_avg)  # Convert to degrees
    
    # Plot actual average joint angle
    ax.plot(time, phi_avg_deg, 'b-', linewidth=2, label=r'Actual $\bar{\phi}$')
    
    # Plot reference average if provided
    if ref_angles is not None and T_ref is not None:
        ref_phi_avg = np.mean(ref_angles, axis=0)
        ref_phi_avg_deg = np.rad2deg(ref_phi_avg)
        ax.plot(T_ref, ref_phi_avg_deg, 'r--', linewidth=1.5, 
               label=r'Reference $\bar{\phi}$', alpha=0.8)
    
    # Mark steering periods with shaded regions
    y_min, y_max = ax.get_ylim()
    for turn_name, turn_config in STEERING_CONFIG.items():
        t_start = turn_config["t_start"]
        t_end = turn_config["t_end"]
        offset_deg = np.rad2deg(turn_config["offset"])
        
        # Check if steering occurs within simulation time
        if t_start <= time[-1]:
            direction = "Right" if offset_deg > 0 else "Left"
            color = 'lightgreen' if offset_deg > 0 else 'lightsalmon'
            
            # Shade the steering region
            ax.axvspan(t_start, min(t_end, time[-1]), alpha=0.3, color=color,
                      label=f'Steer {direction} ({offset_deg:+.0f}°)')
            
            # Add vertical line at start
            ax.axvline(x=t_start, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    # Configure axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, time[-1]//10)))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(max(0.5, time[-1]//20)))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(r'Average Joint Angle $\bar{\phi}$ (degrees)', fontsize=14)
    ax.set_title(r'Average Joint Angle ($\bar{\phi}$) vs Time', fontsize=16)
    ax.set_xlim(time[0], time[-1])
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Average_Joint_Angle.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def plot_tracking_error(time, ref_angles, actual_angles, save=True, show=True):
    """
    Plot joint angle tracking error over time.
    
    Args:
        time: Time array
        ref_angles: Reference angles array (n_joints x time_steps)
        actual_angles: Actual angles array
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    n_joints = min(ref_angles.shape[0], actual_angles.shape[0])
    n_time = min(ref_angles.shape[1], actual_angles.shape[1])
    
    errors = ref_angles[:n_joints, :n_time] - actual_angles[:n_joints, :n_time]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = colormaps['tab10']
    
    for i in range(n_joints):
        ax.plot(time[:n_time], errors[i, :], linewidth=1, color=colors(i),
               label=rf'$e_{{{i+1}}}$', alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tracking Error (rad)', fontsize=12)
    ax.set_title('Joint Angle Tracking Error', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Tracking_Error.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig, ax


def create_summary_plot(time, states, ref_angles, T_ref, save=True, show=True):
    """
    Create a summary figure with multiple subplots.
    
    Args:
        time: Simulation time array
        states: Dictionary with extracted states
        ref_angles: Reference angles array
        T_ref: Reference time array
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Trajectory
    ax2 = fig.add_subplot(2, 2, 2)  # Reference vs Actual (joint 1)
    ax3 = fig.add_subplot(2, 2, 3)  # Position vs time
    ax4 = fig.add_subplot(2, 2, 4)  # Head angle
    
    # 1. Trajectory plot
    ax1.plot(states['px'], states['py'], 'b-', linewidth=1.5)
    ax1.plot(states['px'][0], states['py'][0], 'go', markersize=10, label='Start')
    ax1.plot(states['px'][-1], states['py'][-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('CM Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Joint 1 reference vs actual
    ax2.plot(T_ref, ref_angles[0, :], 'b-', label='Reference', linewidth=1.5)
    ax2.plot(time, states['phi'][0, :], 'r--', label='Actual', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title(r'Joint $\phi_1$ Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    # 3. Position vs time
    ax3.plot(time, states['px'], 'b-', label='X', linewidth=1.5)
    ax3.plot(time, states['py'], 'r-', label='Y', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('CM Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.5)
    
    # 4. Head angle
    ax4.plot(time, np.rad2deg(states['theta_n']), 'g-', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title(r'Head Angle $\theta_N$')
    ax4.grid(True, alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Simulation_Summary.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_performance_metrics(performance, save=True, show=True):
    """
    Create a visualization of simulation performance metrics.
    
    Args:
        performance: Dictionary with performance metrics from run_simulation()
        save: Whether to save figure
        show: Whether to display figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    sim_time = performance['simulation_time']
    wall_time = performance['wall_clock_time']
    rtf = performance['real_time_factor']
    num_links = performance.get('num_links', 'N/A')
    time_steps = performance['time_steps']
    
    # 1. Bar chart: Simulation Time vs Wall-Clock Time
    ax1 = axes[0]
    bars = ax1.bar(['Simulation\nTime', 'Wall-Clock\nTime'], 
                   [sim_time, wall_time], 
                   color=['#2196F3', '#4CAF50'], 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, [sim_time, wall_time]):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Set y-axis to start from 0
    ax1.set_ylim(0, max(sim_time, wall_time) * 1.2)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. Gauge-style RTF visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw RTF as a large number with context
    ax2.text(5, 7, f'{rtf:.0f}x', fontsize=48, fontweight='bold', 
             ha='center', va='center', color='#2196F3')
    ax2.text(5, 4.5, 'Real-Time Factor', fontsize=14, ha='center', va='center')
    
    # Add interpretation
    if rtf > 100:
        status_text = "⚡ Extremely Fast"
        status_color = '#4CAF50'
    elif rtf > 10:
        status_text = "✓ Very Fast"
        status_color = '#8BC34A'
    elif rtf > 1:
        status_text = "✓ Faster than Real-Time"
        status_color = '#CDDC39'
    else:
        status_text = "✗ Slower than Real-Time"
        status_color = '#FF5722'
    
    ax2.text(5, 2.5, status_text, fontsize=14, ha='center', va='center', 
             color=status_color, fontweight='bold')
    
    # Add explanation
    ax2.text(5, 1, f'{sim_time:.1f}s simulated in {wall_time:.4f}s', 
             fontsize=10, ha='center', va='center', style='italic', alpha=0.7)
    
    ax2.set_title('Performance Rating', fontsize=14, fontweight='bold', pad=20)
    
    # 3. Info panel
    ax3 = axes[2]
    ax3.axis('off')
    
    info_text = f"""
    SIMULATION METRICS
    ══════════════════════════
    
    Number of Links:        {num_links}
    Time Steps:             {time_steps}
    Time per Step:          {performance['time_per_step_ms']:.3f} ms
    
    ══════════════════════════
    
    WHAT THE NUMBERS MEAN
    ══════════════════════════
    
    • Simulation Time: The duration of 
      snake motion being simulated
      
    • Wall-Clock Time: Actual computing
      time on your machine
      
    • Real-Time Factor (RTF):
      RTF = Sim Time / Wall Time
      
      RTF > 1  →  Faster than reality
      RTF = 1  →  Real-time
      RTF < 1  →  Slower than reality
    """
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    
    ax3.set_title('Information', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('Performance_Metrics.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def compute_link_positions(phi, theta_n, px, py, l, N):
    """
    Compute the positions of all link centers and endpoints from state variables.
    
    The snake robot kinematics:
    - theta = H * phi_bar, where phi_bar = [phi_1, ..., phi_{N-1}, theta_N]
    - Each link i has center at (x_i, y_i) and orientation theta_i
    
    Args:
        phi: Joint angles array (N-1,) in radians
        theta_n: Head angle (scalar) in radians
        px: CM x position (scalar)
        py: CM y position (scalar)
        l: Link length (full length = 2*l where l is half-length)
        N: Number of links
        
    Returns:
        link_x: Array of link center x positions (N,)
        link_y: Array of link center y positions (N,)
        theta: Array of link angles (N,)
    """
    # Build phi_bar = [phi_1, ..., phi_{N-1}, theta_N]
    phi_bar = np.append(phi, theta_n)
    
    # Build H matrix for theta = H * phi_bar
    H = -1 * np.triu(np.ones((N, N)))
    H[:, -1] = -1 * H[:, -1]
    
    # Compute absolute link angles
    theta = H @ phi_bar
    
    # Compute link center positions from CM position
    # Using the kinematic relationships
    # The CM is the average of all link centers
    
    # K matrix for position calculation (from textbook)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j < i:
                K[i, j] = 1
            elif j == i:
                K[i, j] = 0.5
            else:
                K[i, j] = 0
    
    # Deviation from mean
    K_bar = K - np.mean(K, axis=0)
    
    # Link center positions
    link_x = px + l * K_bar @ np.cos(theta)
    link_y = py + l * K_bar @ np.sin(theta)
    
    return link_x, link_y, theta


def animate_snake_robot(time, states, phy_props, save=True, show=True, 
                        speed_factor=1.0, interval=50, trail_length=50):
    """
    Create an animation of the snake robot movement.
    
    Args:
        time: Time array from simulation
        states: States dictionary from extract_states()
        phy_props: Physical properties dictionary
        save: Whether to save animation as GIF/MP4
        show: Whether to display animation
        speed_factor: Speed up/slow down factor (1.0 = real-time playback)
        interval: Milliseconds between frames (lower = faster)
        trail_length: Number of past CM positions to show as trail
        
    Returns:
        anim: Animation object
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import FancyBboxPatch, Circle
    import matplotlib.patches as mpatches
    
    setup_plot_style()
    
    # Extract parameters
    N = phy_props['N']
    l = phy_props['l']  # This is half-link length
    
    # Extract state arrays
    phi = states['phi']          # (N-1, time_steps)
    theta_n = states['theta_n']  # (time_steps,)
    px = states['px']            # (time_steps,)
    py = states['py']            # (time_steps,)
    
    # Subsample for smoother animation (every nth frame)
    n_frames = len(time)
    skip = max(1, n_frames // 500)  # Limit to ~500 frames
    frame_indices = np.arange(0, n_frames, skip)
    
    # Compute all link positions for all time steps
    print(f"Computing link positions for {len(frame_indices)} frames...")
    all_link_x = []
    all_link_y = []
    all_theta = []
    
    for idx in frame_indices:
        link_x, link_y, theta = compute_link_positions(
            phi[:, idx], theta_n[idx], px[idx], py[idx], l, N
        )
        all_link_x.append(link_x)
        all_link_y.append(link_y)
        all_theta.append(theta)
    
    all_link_x = np.array(all_link_x)
    all_link_y = np.array(all_link_y)
    all_theta = np.array(all_theta)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate bounds with padding
    x_min = np.min(all_link_x) - 0.3
    x_max = np.max(all_link_x) + 0.3
    y_min = np.min(all_link_y) - 0.3
    y_max = np.max(all_link_y) + 0.3
    
    # Ensure equal aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        y_center = (y_max + y_min) / 2
        y_min = y_center - x_range / 2
        y_max = y_center + x_range / 2
    else:
        x_center = (x_max + x_min) / 2
        x_min = x_center - y_range / 2
        x_max = x_center + y_range / 2
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    # Color map for links (gradient from tail to head)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, N))
    
    # Initialize plot elements
    # Links as thick lines
    link_lines = []
    for i in range(N):
        line, = ax.plot([], [], 'o-', linewidth=8, markersize=4,
                       color=colors[i], solid_capstyle='round')
        link_lines.append(line)
    
    # Head marker (circle)
    head_marker, = ax.plot([], [], 'ro', markersize=15, zorder=10,
                          markeredgecolor='darkred', markeredgewidth=2)
    
    # Tail marker (circle)
    tail_marker, = ax.plot([], [], 'go', markersize=12, zorder=10,
                          markeredgecolor='darkgreen', markeredgewidth=2)
    
    # CM trail
    trail_line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.5, label='CM Trail')
    
    # CM current position
    cm_marker, = ax.plot([], [], 'b*', markersize=12, zorder=9, label='CM')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Info text
    info_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Title
    title = ax.set_title(f'Snake Robot Animation ({N} Links)', fontsize=14, fontweight='bold')
    
    def init():
        """Initialize animation."""
        for line in link_lines:
            line.set_data([], [])
        head_marker.set_data([], [])
        tail_marker.set_data([], [])
        trail_line.set_data([], [])
        cm_marker.set_data([], [])
        time_text.set_text('')
        info_text.set_text('')
        return link_lines + [head_marker, tail_marker, trail_line, cm_marker, time_text, info_text]
    
    def update(frame):
        """Update animation frame."""
        idx = frame
        t = time[frame_indices[idx]]
        
        # Get link positions for this frame
        link_x = all_link_x[idx]
        link_y = all_link_y[idx]
        theta = all_theta[idx]
        
        # Update each link as a line from one end to the other
        for i in range(N):
            # Link endpoints
            x_start = link_x[i] - l * np.cos(theta[i])
            x_end = link_x[i] + l * np.cos(theta[i])
            y_start = link_y[i] - l * np.sin(theta[i])
            y_end = link_y[i] + l * np.sin(theta[i])
            link_lines[i].set_data([x_start, x_end], [y_start, y_end])
        
        # Head position (front of last link)
        head_x = link_x[-1] + l * np.cos(theta[-1])
        head_y = link_y[-1] + l * np.sin(theta[-1])
        head_marker.set_data([head_x], [head_y])
        
        # Tail position (back of first link)
        tail_x = link_x[0] - l * np.cos(theta[0])
        tail_y = link_y[0] - l * np.sin(theta[0])
        tail_marker.set_data([tail_x], [tail_y])
        
        # CM trail
        trail_start = max(0, idx - trail_length)
        trail_x = px[frame_indices[trail_start:idx+1]]
        trail_y = py[frame_indices[trail_start:idx+1]]
        trail_line.set_data(trail_x, trail_y)
        
        # CM marker
        cm_marker.set_data([px[frame_indices[idx]]], [py[frame_indices[idx]]])
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f} s')
        
        # Update info text
        cm_x = px[frame_indices[idx]]
        cm_y = py[frame_indices[idx]]
        head_angle = np.rad2deg(theta_n[frame_indices[idx]])
        info_text.set_text(f'CM: ({cm_x:.3f}, {cm_y:.3f}) m\nHead angle: {head_angle:.1f}°')
        
        return link_lines + [head_marker, tail_marker, trail_line, cm_marker, time_text, info_text]
    
    # Create animation
    print(f"Creating animation with {len(frame_indices)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                        init_func=init, blit=True, interval=interval)
    
    # Save animation
    if save:
        try:
            # Try to save as GIF (requires pillow)
            print("Saving animation as GIF...")
            anim.save('snake_robot_animation.gif', writer='pillow', fps=20, dpi=100)
            print("Animation saved as 'snake_robot_animation.gif'")
        except Exception as e:
            print(f"Could not save GIF: {e}")
            try:
                # Try to save as MP4 (requires ffmpeg)
                print("Trying to save as MP4...")
                anim.save('snake_robot_animation.mp4', writer='ffmpeg', fps=20, dpi=100)
                print("Animation saved as 'snake_robot_animation.mp4'")
            except Exception as e2:
                print(f"Could not save MP4: {e2}")
                print("To save animations, install: pip install pillow")
    
    if show:
        plt.show()
    
    return anim


def create_snake_frames(time, states, phy_props, n_frames=6, save=True, show=True):
    """
    Create a multi-panel figure showing snake robot at different time points.
    
    This is useful for publications or when animation is not possible.
    
    Args:
        time: Time array from simulation
        states: States dictionary from extract_states()
        phy_props: Physical properties dictionary
        n_frames: Number of frames to show
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        fig, axes: Figure and axes objects
    """
    setup_plot_style()
    
    # Extract parameters
    N = phy_props['N']
    l = phy_props['l']
    
    # Extract state arrays
    phi = states['phi']
    theta_n = states['theta_n']
    px = states['px']
    py = states['py']
    
    # Select time indices to show
    n_steps = len(time)
    indices = np.linspace(0, n_steps - 1, n_frames).astype(int)
    
    # Create subplots
    n_cols = min(3, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color map for links
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, N))
    
    # Compute global bounds
    all_x, all_y = [], []
    for idx in indices:
        link_x, link_y, theta = compute_link_positions(
            phi[:, idx], theta_n[idx], px[idx], py[idx], l, N
        )
        all_x.extend(link_x)
        all_y.extend(link_y)
    
    x_margin = (max(all_x) - min(all_x)) * 0.2 + 0.1
    y_margin = (max(all_y) - min(all_y)) * 0.2 + 0.1
    x_lim = (min(all_x) - x_margin, max(all_x) + x_margin)
    y_lim = (min(all_y) - y_margin, max(all_y) + y_margin)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        t = time[idx]
        
        # Compute link positions
        link_x, link_y, theta = compute_link_positions(
            phi[:, idx], theta_n[idx], px[idx], py[idx], l, N
        )
        
        # Plot CM trail up to this point
        ax.plot(px[:idx+1], py[:idx+1], 'b-', linewidth=1, alpha=0.3)
        
        # Plot each link
        for j in range(N):
            x_start = link_x[j] - l * np.cos(theta[j])
            x_end = link_x[j] + l * np.cos(theta[j])
            y_start = link_y[j] - l * np.sin(theta[j])
            y_end = link_y[j] + l * np.sin(theta[j])
            ax.plot([x_start, x_end], [y_start, y_end], 'o-', 
                   linewidth=6, markersize=3, color=colors[j], solid_capstyle='round')
        
        # Mark head and tail
        head_x = link_x[-1] + l * np.cos(theta[-1])
        head_y = link_y[-1] + l * np.sin(theta[-1])
        tail_x = link_x[0] - l * np.cos(theta[0])
        tail_y = link_y[0] - l * np.sin(theta[0])
        
        ax.plot(head_x, head_y, 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=1.5)
        ax.plot(tail_x, tail_y, 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=1.5)
        ax.plot(px[idx], py[idx], 'b*', markersize=10)  # CM
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f't = {t:.2f} s', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
    
    # Hide unused axes
    for i in range(n_frames, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Snake Robot Motion Sequence ({N} Links)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig('Snake_Motion_Sequence.png', dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        print("Saved 'Snake_Motion_Sequence.png'")
    
    if show:
        plt.show()
    
    return fig, axes


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing plotting module...")
    
    t = np.linspace(0, 10, 100)
    px = t * 0.1 + 0.1 * np.sin(t)
    py = 0.05 * np.sin(2 * t)
    
    plot_cm_trajectory(px, py, time=t, save=False, show=True)
    print("Plotting test complete!")
