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
    ax.plot(px[-1], py[-1], 'rs', markersize=14, label='End', zorder=5,
            markeredgecolor='darkred', markeredgewidth=2)
    
    # Add time markers (small dots at specific times)
    if time is not None and len(time) > 10:
        # Show markers at regular time intervals
        t_max = time[-1]
        marker_interval = max(1, int(t_max / 5))  # ~5 markers
        time_markers = np.arange(marker_interval, t_max, marker_interval)
        for t_mark in time_markers:
            idx = np.argmin(np.abs(time - t_mark))
            if idx < len(px) and idx > 0:
                ax.plot(px[idx], py[idx], 'ko', markersize=6, zorder=4)
                ax.annotate(f'{t_mark:.0f}s', (px[idx], py[idx]), 
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=9, alpha=0.8)
    
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.set_title('Snake Robot Center of Mass Trajectory', fontsize=16)
    ax.legend(loc='best', fontsize=12)
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


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing plotting module...")
    
    t = np.linspace(0, 10, 100)
    px = t * 0.1 + 0.1 * np.sin(t)
    py = 0.05 * np.sin(2 * t)
    
    plot_cm_trajectory(px, py, time=t, save=False, show=True)
    print("Plotting test complete!")
