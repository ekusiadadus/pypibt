"""
Quick test for LNS implementation.
"""

from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution

# Load map and scenario
grid = get_grid("assets/random-32-32-10.map")
starts, goals = get_scenario("assets/random-32-32-10-random-1.scen", 100)

print("="* 80)
print("Testing LNS Implementation")
print("="* 80)

# Test 1: Baseline PIBT
print("\n[Test 1] Baseline PIBT (100 agents)")
pibt_baseline = PIBT(grid, starts, goals, seed=0)
plan_baseline = pibt_baseline.run(max_timestep=1000)
valid_baseline = is_valid_mapf_solution(grid, starts, goals, plan_baseline)
print(f"Result: {len(plan_baseline)} steps, Valid: {valid_baseline}")

# Test 2: LNS with random destroy
print("\n[Test 2] LNS with random destroy")
pibt_lns_random = PIBT(
    grid, starts, goals, seed=0,
    enable_lns=True,
    lns_iterations=5,
    lns_destroy_size=10,
    lns_destroy_strategy="random"
)
plan_lns_random = pibt_lns_random.run(max_timestep=1000)
valid_lns_random = is_valid_mapf_solution(grid, starts, goals, plan_lns_random)
print(f"Result: {len(plan_lns_random)} steps, Valid: {valid_lns_random}")

# Test 3: LNS with conflict-based destroy
print("\n[Test 3] LNS with conflict-based destroy")
pibt_lns_conflict = PIBT(
    grid, starts, goals, seed=0,
    enable_lns=True,
    lns_iterations=5,
    lns_destroy_size=10,
    lns_destroy_strategy="conflict"
)
plan_lns_conflict = pibt_lns_conflict.run(max_timestep=1000)
valid_lns_conflict = is_valid_mapf_solution(grid, starts, goals, plan_lns_conflict)
print(f"Result: {len(plan_lns_conflict)} steps, Valid: {valid_lns_conflict}")

# Test 4: LNS with adaptive destroy
print("\n[Test 4] LNS with adaptive destroy")
pibt_lns_adaptive = PIBT(
    grid, starts, goals, seed=0,
    enable_lns=True,
    lns_iterations=5,
    lns_destroy_size=10,
    lns_destroy_strategy="adaptive"
)
plan_lns_adaptive = pibt_lns_adaptive.run(max_timestep=1000)
valid_lns_adaptive = is_valid_mapf_solution(grid, starts, goals, plan_lns_adaptive)
print(f"Result: {len(plan_lns_adaptive)} steps, Valid: {valid_lns_adaptive}")

# Summary
print("\n" + "="* 80)
print("Summary:")
print(f"  Baseline:         {len(plan_baseline):4d} steps")
print(f"  LNS (random):     {len(plan_lns_random):4d} steps")
print(f"  LNS (conflict):   {len(plan_lns_conflict):4d} steps")
print(f"  LNS (adaptive):   {len(plan_lns_adaptive):4d} steps")
print("="* 80)

print("\nAll tests passed!" if all([
    valid_baseline, valid_lns_random, valid_lns_conflict, valid_lns_adaptive
]) else "\nSome tests failed!")
