"""
Test Priority Learning integration with PIBT.
"""

from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution
import time

print("="* 80)
print("Testing Priority Learning")
print("="* 80)

grid = get_grid("assets/random-32-32-10.map")
starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 100)

# Test 1: Baseline
print("\n[Test 1] Baseline PIBT")
start_time = time.time()
pibt_baseline = PIBT(grid, starts, goals, seed=0)
plan_baseline = pibt_baseline.run(max_timestep=1000)
time_baseline = time.time() - start_time
valid_baseline = is_valid_mapf_solution(grid, starts, goals, plan_baseline)
print(f"Result: {len(plan_baseline)} steps, {time_baseline:.3f}s, Valid: {valid_baseline}")

# Test 2: Priority Learning
print("\n[Test 2] PIBT with Priority Learning")
start_time = time.time()
pibt_learned = PIBT(
    grid, starts, goals, seed=0,
    enable_priority_learning=True,
    priority_model_path="models/priority_network.pth"
)
plan_learned = pibt_learned.run(max_timestep=1000)
time_learned = time.time() - start_time
valid_learned = is_valid_mapf_solution(grid, starts, goals, plan_learned)
print(f"Result: {len(plan_learned)} steps, {time_learned:.3f}s, Valid: {valid_learned}")

# Test 3: Priority Learning + Hindrance
print("\n[Test 3] Priority Learning + Hindrance")
start_time = time.time()
pibt_combined = PIBT(
    grid, starts, goals, seed=0,
    enable_priority_learning=True,
    priority_model_path="models/priority_network.pth",
    enable_hindrance=True,
    hindrance_weight=0.3
)
plan_combined = pibt_combined.run(max_timestep=1000)
time_combined = time.time() - start_time
valid_combined = is_valid_mapf_solution(grid, starts, goals, plan_combined)
print(f"Result: {len(plan_combined)} steps, {time_combined:.3f}s, Valid: {valid_combined}")

# Summary
print("\n" + "="* 80)
print("Summary:")
print(f"  Baseline:             {len(plan_baseline):4d} steps | {time_baseline:.3f}s")
print(f"  Priority Learning:    {len(plan_learned):4d} steps | {time_learned:.3f}s")
print(f"  PL + Hindrance:       {len(plan_combined):4d} steps | {time_combined:.3f}s")

if len(plan_learned) < len(plan_baseline):
    improvement = (len(plan_baseline) - len(plan_learned)) / len(plan_baseline) * 100
    print(f"\n  Priority Learning improvement: +{improvement:.2f}%")
else:
    print(f"\n  No improvement from Priority Learning")

print("="* 80)

print("\nAll tests passed!" if all([valid_baseline, valid_learned, valid_combined]) else "\nSome tests failed!")
