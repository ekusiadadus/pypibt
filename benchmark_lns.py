"""
Comprehensive benchmark for LNS implementation.
Compares all optimization methods including the new LNS.
"""

import time
from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution

# Test scenarios
scenarios = [
    {"name": "Small (100 agents)", "agents": 100},
    {"name": "Medium (200 agents)", "agents": 200},
    {"name": "Large (400 agents)", "agents": 400},
]

# Algorithm variants
variants = [
    {
        "name": "Baseline",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": False,
        },
    },
    {
        "name": "Hindrance + Regret",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": True,
            "regret_learning_iterations": 3,
            "regret_weight": 0.2,
            "enable_anytime": False,
            "enable_lns": False,
        },
    },
    {
        "name": "Anytime PIBT",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": True,
            "anytime_time_limit_ms": 500,
            "anytime_beam_width": 5,
            "enable_lns": False,
        },
    },
    {
        "name": "LNS (Random)",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "random",
        },
    },
    {
        "name": "LNS (Conflict)",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "conflict",
        },
    },
    {
        "name": "LNS (Adaptive)",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "adaptive",
        },
    },
    {
        "name": "LNS + Hindrance",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "adaptive",
        },
    },
    {
        "name": "Full (LNS + All)",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,  # Disable for speed
            "enable_anytime": False,
            "enable_lns": True,
            "lns_iterations": 10,
            "lns_destroy_size": 20,
            "lns_destroy_strategy": "adaptive",
        },
    },
]

print("=" * 100)
print("LNS BENCHMARK - Phase 5")
print("=" * 100)
print()

grid = get_grid("assets/random-32-32-10.map")

for scenario in scenarios:
    print(f"\\n{'=' * 100}")
    print(f"Scenario: {scenario['name']}")
    print(f"{'=' * 100}\\n")

    starts, goals = get_scenario(
        "assets/random-32-32-10-random-1.scen", scenario["agents"]
    )

    results = []

    for variant in variants:
        print(f"  {variant['name']:30s}...", end=" ", flush=True)

        try:
            start_time = time.time()
            pibt = PIBT(grid, starts, goals, seed=0, **variant["params"])
            plan = pibt.run(max_timestep=2000)
            elapsed = time.time() - start_time

            valid = is_valid_mapf_solution(grid, starts, goals, plan)
            timesteps = len(plan) if valid else float("inf")

            results.append(
                {
                    "name": variant["name"],
                    "timesteps": timesteps,
                    "time": elapsed,
                    "valid": valid,
                }
            )

            if valid:
                print(f"{timesteps:4d} steps | {elapsed:7.3f}s")
            else:
                print("INVALID")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append(
                {
                    "name": variant["name"],
                    "timesteps": float("inf"),
                    "time": 0,
                    "valid": False,
                }
            )

    # Summary
    print("\\n  Summary:")
    baseline = next((r for r in results if r["name"] == "Baseline"), None)

    if baseline and baseline["valid"]:
        for result in results:
            if result["valid"] and baseline["timesteps"] != float("inf"):
                improvement = (
                    (baseline["timesteps"] - result["timesteps"])
                    / baseline["timesteps"]
                    * 100
                )
                print(
                    f"    {result['name']:30s}: {result['timesteps']:4.0f} steps "
                    f"({improvement:+6.2f}%) | {result['time']:7.3f}s"
                )

print("\\n" + "=" * 100)
print("BENCHMARK COMPLETE")
print("=" * 100)

print("""
Key Findings:
1. LNS: Iterative improvement via destroy-and-repair
2. LNS + Hindrance: Combines local search with global coordination
3. Adaptive strategy: Dynamically switches between exploration and exploitation

All LNS variants based on MAPF-LNS2 (AAAI 2022) and LNS2+RL (2024-2025 research).
""")

print("Benchmark complete!")
