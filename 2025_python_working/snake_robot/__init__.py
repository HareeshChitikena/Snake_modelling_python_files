"""
Snake Robot Package
===================
A modular implementation of snake robot dynamics and simulation.

Modules:
- config: Configuration parameters
- snake_init: Initialization and state setup
- dynamic_model: Dynamics and ODE solver
- plotting: Visualization functions
"""

from .config import (
    PHYSICAL_PROPERTIES,
    SNAKE_MOTION_PARAMS,
    SIMULATION_SETTINGS,
    CONTROL_GAINS,
    INITIAL_CONDITIONS,
    get_snake_parameters,
    get_all_config
)

from .snake_init import SnakeInitializer
from .dynamic_model import SnakeDynamicModel
from . import plotting

__version__ = "1.0.0"
__author__ = "Snake Robot Project"

__all__ = [
    'PHYSICAL_PROPERTIES',
    'SNAKE_MOTION_PARAMS', 
    'SIMULATION_SETTINGS',
    'CONTROL_GAINS',
    'INITIAL_CONDITIONS',
    'get_snake_parameters',
    'get_all_config',
    'SnakeInitializer',
    'SnakeDynamicModel',
    'plotting'
]
