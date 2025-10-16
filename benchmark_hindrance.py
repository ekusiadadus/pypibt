"""Benchmark script to test different hindrance weight values."""

from pypibt import PIBT, get_grid, get_scenario

# Load test scenario
grid = get_grid("assets/random-32-32-10.map")
starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 200)

print("=== Hindrance Weight Benchmark ===\n")
print("Testing 200 agents on random-32-32-10 map\n")

weights_to_test = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

results = []

for weight in weights_to_test:
    pibt = PIBT(grid, starts, goals, seed=0, hindrance_weight=weight)
    plan = pibt.run(max_timestep=1000)
    timesteps = len(plan)
    results.append((weight, timesteps))
    print(f"hindrance_weight={weight:.2f}: {timesteps} timesteps")

print("\n=== Summary ===")
baseline = results[0][1]  # weight=0.0 is baseline
best_weight, best_timesteps = min(results, key=lambda x: x[1])

print(f"Baseline (weight=0.0): {baseline} timesteps")
print(f"Best result: weight={best_weight:.2f}, {best_timesteps} timesteps")
improvement = ((baseline - best_timesteps) / baseline) * 100
print(f"Improvement: {improvement:.2f}%")
