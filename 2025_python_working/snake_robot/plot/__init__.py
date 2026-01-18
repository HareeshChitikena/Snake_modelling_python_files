"""
Plot Package
============
Contains visualization and plotting functions for snake robot simulation.
"""

from .plotting import (
    plot_reference_angles,
    plot_reference_vs_actual,
    plot_cm_trajectory,
    plot_position_vs_time,
    plot_velocity_vs_time,
    plot_head_angle,
    plot_average_joint_angle,
    plot_tracking_error,
    create_summary_plot,
    plot_performance_metrics,
    animate_snake_robot,
    create_snake_frames,
    copy_outputs_to_final
)

__all__ = [
    'plot_reference_angles',
    'plot_reference_vs_actual',
    'plot_cm_trajectory',
    'plot_position_vs_time',
    'plot_velocity_vs_time',
    'plot_head_angle',
    'plot_average_joint_angle',
    'plot_tracking_error',
    'create_summary_plot',
    'plot_performance_metrics',
    'animate_snake_robot',
    'create_snake_frames',
    'copy_outputs_to_final'
]
