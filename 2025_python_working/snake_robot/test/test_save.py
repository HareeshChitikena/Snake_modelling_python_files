#!/usr/bin/env python
"""Test script to verify plotting and saving functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from main import run_simulation, create_plots
from plot import plotting
import numpy as np

print("Starting test...")

# Run short simulation
results = run_simulation(num_links=10, simulation_time=2, accuracy='1', verbose=False)

print(f"\nOutput directory: {plotting.OUTPUT_DIR}")
print(f"Temp plots dir: {plotting._TEMP_PLOTS}")

# Check data shapes
T = results['time']
T_ref = results['T_ref']
ref_angles = results['ref_angles']
states = results['states']

print(f"\nData shapes:")
print(f"  ref_angles: {ref_angles.shape}")
print(f"  states['phi']: {states['phi'].shape}")
print(f"  T: {T.shape}")
print(f"  T_ref: {T_ref.shape}")

# Test saving
print("\nTesting plot_reference_vs_actual...")
try:
    plotting.plot_reference_vs_actual(ref_angles, T_ref, states['phi'], T, 
                                      save=True, show=False)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Copy to final
plotting.copy_outputs_to_final()

print("\nTest complete!")
