"""Quick speed test for the snake robot simulation."""
import time
from config import update_config, SIMULATION_SETTINGS
from snake_init import SnakeInitializer
from dynamic_model import SnakeDynamicModel

# Test configuration - small test first
num_links = 5
sim_time = 10  # Reduced for quick test
accuracy = "1"  # low

print("=" * 50)
print("Snake Robot Speed Test (OPTIMIZED)")
print("=" * 50)

# Update config
update_config(num_links, sim_time, accuracy)
print(f"Links: {num_links}, Simulation time: {sim_time}s")
print(f"Solver: {SIMULATION_SETTINGS['solver_method']}")
print(f"Tolerances: rtol={SIMULATION_SETTINGS['rtol']}, atol={SIMULATION_SETTINGS['atol']}")

# Initialize
print("\nInitializing...")
t0 = time.time()
snake = SnakeInitializer()
phy_props, init_vals, snake_params, control_gains = snake.get_parameters()
print(f"  Init time: {time.time()-t0:.3f}s")

# Create model
print("Creating model...")
t0 = time.time()
model = SnakeDynamicModel(phy_props, init_vals, snake_params, control_gains)
print(f"  Model time: {time.time()-t0:.3f}s")

# Generate reference
print("Generating reference angles...")
t0 = time.time()
model.generate_reference_angles()
print(f"  Reference time: {time.time()-t0:.3f}s")

# Run simulation
print("\nRunning simulation...")
t0 = time.time()
T, q = model.simulate(verbose=True)
sim_time_taken = time.time() - t0
print(f"\n  TOTAL Simulation time: {sim_time_taken:.2f}s")
print(f"  Time points: {len(T)}")
print(f"  Speed ratio: {sim_time/sim_time_taken:.1f}x realtime")

print("\n" + "=" * 50)
print("Test complete!")
print("=" * 50)
