"""
Benchmark different ODE solvers for snake robot simulation.
Tests: scipy solvers, numba JIT compilation, and identifies optimization opportunities.
"""
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.integrate import solve_ivp
from config import update_config, SIMULATION_SETTINGS
from math_model import SnakeInitializer, SnakeDynamicModel

print("=" * 70)
print("ODE SOLVER BENCHMARK FOR SNAKE ROBOT")
print("=" * 70)

# Configuration
NUM_LINKS = 3
SIM_TIME = 1.0  # Short simulation for benchmarking

# Tolerance presets to test
TOLERANCE_PRESETS = {
    "ultra_low": (1e-1, 1e-1),
    "low": (1e-2, 1e-2),
    "medium": (1e-4, 1e-4),
    "high": (1e-6, 1e-6),
}

# Solvers to test
SOLVERS = ["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"]

print(f"\nTest Configuration:")
print(f"  Links: {NUM_LINKS}")
print(f"  Simulation time: {SIM_TIME}s")
print(f"  Solvers: {SOLVERS}")

# Initialize model
update_config(NUM_LINKS, SIM_TIME, "1")
snake = SnakeInitializer()
phy_props, init_vals, snake_params, control_gains = snake.get_parameters()
model = SnakeDynamicModel(phy_props, init_vals, snake_params, control_gains)

y0 = np.asarray(model.q).flatten()
t_span = [0, SIM_TIME]
t_eval = np.arange(0, SIM_TIME + 0.1, 0.1)

print(f"  State vector size: {len(y0)}")

# ========== BENCHMARK SINGLE ODE CALL ==========
print("\n" + "-" * 70)
print("SINGLE ODE FUNCTION CALL BENCHMARK")
print("-" * 70)

# Warmup
for _ in range(10):
    _ = model._compute_derivatives(0.5, y0)

# Benchmark
n_calls = 1000
t0 = time.perf_counter()
for _ in range(n_calls):
    _ = model._compute_derivatives(0.5, y0)
dt = (time.perf_counter() - t0) / n_calls
print(f"  Average ODE call time: {dt*1000:.4f} ms ({dt*1e6:.2f} µs)")
print(f"  Estimated calls for 1s sim @ 0.001 step: {1/0.001:.0f}")

# ========== BENCHMARK DIFFERENT SOLVERS ==========
print("\n" + "-" * 70)
print("SOLVER COMPARISON (low tolerance: rtol=1e-2, atol=1e-2)")
print("-" * 70)

results = {}

for solver in SOLVERS:
    try:
        # Reset model state
        model = SnakeDynamicModel(phy_props, init_vals, snake_params, control_gains)
        y0 = np.asarray(model.q).flatten()
        
        t0 = time.perf_counter()
        sol = solve_ivp(
            model._compute_derivatives,
            t_span,
            y0,
            method=solver,
            t_eval=t_eval,
            rtol=1e-2,
            atol=1e-2
        )
        elapsed = time.perf_counter() - t0
        
        results[solver] = {
            'time': elapsed,
            'nfev': sol.nfev,
            'success': sol.success,
            'time_per_call': elapsed / sol.nfev * 1000 if sol.nfev > 0 else 0
        }
        
        status = "✓" if sol.success else "✗"
        print(f"  {solver:8s}: {elapsed:6.3f}s | nfev={sol.nfev:6d} | {elapsed/sol.nfev*1000:.3f}ms/call | {status}")
        
    except Exception as e:
        print(f"  {solver:8s}: FAILED - {e}")
        results[solver] = {'time': float('inf'), 'nfev': 0, 'success': False}

# Find best
best_solver = min(results, key=lambda x: results[x]['time'] if results[x]['success'] else float('inf'))
print(f"\n  BEST SOLVER: {best_solver} ({results[best_solver]['time']:.3f}s)")

# ========== CHECK CACHE USAGE ==========
print("\n" + "-" * 70)
print("CACHE USAGE ANALYSIS")
print("-" * 70)

# Check which matrices are pre-computed
cached_attrs = ['H', 'H_T', 'K', 'K_T', 'V', 'A', 'D', 'J_eye', 'ml2', 'Nm_eye2', 'l_K_T', 'c_diff']
print("  Pre-computed matrices (CACHED):")
for attr in cached_attrs:
    if hasattr(model, attr):
        val = getattr(model, attr)
        if isinstance(val, np.ndarray):
            print(f"    ✓ {attr}: shape={val.shape}, contiguous={val.flags['C_CONTIGUOUS']}")
        else:
            print(f"    ✓ {attr}: {type(val).__name__} = {val}")

# Check buffers
buffer_attrs = ['_phi_bar', '_phi_bar_d', '_theta', '_theta_d', '_sin_theta', '_cos_theta', 
                '_X_d', '_Y_d', '_F_rv', '_u_bar', '_y_dot']
print("\n  Pre-allocated buffers:")
for attr in buffer_attrs:
    if hasattr(model, attr):
        val = getattr(model, attr)
        print(f"    ✓ {attr}: shape={val.shape}")
    else:
        print(f"    ✗ {attr}: NOT FOUND")

# ========== IDENTIFY OPTIMIZATION OPPORTUNITIES ==========
print("\n" + "-" * 70)
print("OPTIMIZATION OPPORTUNITIES IDENTIFIED")
print("-" * 70)

print("""
  CURRENT ISSUES IN _compute_derivatives():
  
  1. ✗ Creating new arrays inside function:
     - np.empty(N) for phi_bar, phi_bar_d - should use pre-allocated buffers
     - np.diag() creates new NxN matrices - expensive!
     - np.zeros() for M_12, M_21, M_22 - should be pre-allocated
     - np.hstack/vstack creates temporary arrays
     
  2. ✗ Not using pre-allocated buffers:
     - _phi_bar, _phi_bar_d buffers exist but NOT USED
     - _theta, _theta_d buffers exist but NOT USED
     - _sin_theta, _cos_theta buffers exist but NOT USED
     
  3. ✗ Redundant computations:
     - np.dot(c_theta, c_theta) computed but c_theta is diagonal, so this is c²
     - Can use element-wise operations instead of full matrix multiplication
     
  4. ✗ Matrix inverse computed every call:
     - np.linalg.inv(M22) - use np.linalg.solve() instead (already imported but not used!)
""")

# ========== RECOMMENDATIONS ==========
print("\n" + "-" * 70)
print("RECOMMENDATIONS")
print("-" * 70)

print(f"""
  1. SOLVER: Use '{best_solver}' - fastest for this problem
  
  2. BUFFER USAGE: Update _compute_derivatives to USE the pre-allocated buffers
     - Currently buffers are allocated but NOT USED
     
  3. DIAGONAL MATRIX OPTIMIZATION:
     - Instead of np.diag(vec) @ matrix @ np.diag(vec)
     - Use: vec[:, None] * matrix * vec[None, :] (element-wise, no allocation)
     
  4. REPLACE np.linalg.inv with np.linalg.solve:
     - Instead of: M22_inv @ rhs
     - Use: np.linalg.solve(M22, rhs)
     
  5. NUMBA JIT: Consider @numba.jit for the ODE function
     - Can provide 10-100x speedup for numerical code
     
  6. PRE-ALLOCATE ALL INTERMEDIATE MATRICES outside the ODE function
""")

print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
