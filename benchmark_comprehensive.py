"""
Comprehensive benchmark comparing all PIBT optimizations.
Tests across multiple scenarios and agent counts.
"""

import time
from pypibt import PIBT, get_grid, get_scenario, is_valid_mapf_solution

# Test configurations
test_configs = [
    {
        "name": "Small (100 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 100,
    },
    {
        "name": "Medium (200 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 200,
    },
    {
        "name": "Large (400 agents)",
        "map": "assets/random-32-32-10.map",
        "scen": "assets/random-32-32-10-random-1.scen",
        "num_agents": 400,
    },
]

# Algorithm variants to test
variants = [
    {
        "name": "Baseline",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "name": "Hindrance",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.3,
            "enable_regret_learning": False,
            "priority_increment": 1.0,
        },
    },
    {
        "name": "Regret Learning",
        "params": {
            "enable_hindrance": False,
            "enable_regret_learning": True,
            "regret_learning_iterations": 3,
            "regret_weight": 0.2,
            "priority_increment": 1.0,
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
            "priority_increment": 1.0,
        },
    },
    {
        "name": "Optuna Best",
        "params": {
            "enable_hindrance": True,
            "hindrance_weight": 0.31,
            "enable_regret_learning": False,
            "priority_increment": 0.59,
        },
    },
]

print("=" * 80)
print("COMPREHENSIVE PIBT BENCHMARK")
print("=" * 80)
print()

results = {}

for config in test_configs:
    print(f"\n{'=' * 80}")
    print(f"Test: {config['name']}")
    print(f"{'=' * 80}\n")

    grid = get_grid(config["map"])
    starts, goals = get_scenario(config["scen"], config["num_agents"])

    config_results = []

    for variant in variants:
        print(f"  Testing {variant['name']:20s}...", end=" ", flush=True)

        try:
            start_time = time.time()

            pibt = PIBT(grid, starts, goals, seed=0, **variant["params"])
            plan = pibt.run(max_timestep=2000)

            elapsed = time.time() - start_time

            valid = is_valid_mapf_solution(grid, starts, goals, plan)
            timesteps = len(plan) if valid else float("inf")

            config_results.append(
                {
                    "variant": variant["name"],
                    "timesteps": timesteps,
                    "elapsed": elapsed,
                    "valid": valid,
                }
            )

            if valid:
                print(f"{timesteps:4d} steps | {elapsed:6.2f}s")
            else:
                print("INVALID SOLUTION")

        except Exception as e:
            print(f"ERROR: {e}")
            config_results.append(
                {
                    "variant": variant["name"],
                    "timesteps": float("inf"),
                    "elapsed": 0,
                    "valid": False,
                }
            )

    results[config["name"]] = config_results

    # Print summary for this configuration
    print("\n  Summary:")
    baseline_timesteps = next(
        (r["timesteps"] for r in config_results if r["variant"] == "Baseline"), None
    )

    for result in config_results:
        if result["valid"] and baseline_timesteps and baseline_timesteps != float("inf"):
            improvement = (
                (baseline_timesteps - result["timesteps"]) / baseline_timesteps * 100
            )
            print(
                f"    {result['variant']:20s}: {result['timesteps']:4.0f} steps "
                f"({improvement:+.1f}%) | {result['elapsed']:.2f}s"
            )

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print()

for config_name, config_results in results.items():
    print(f"{config_name}:")
    baseline = next((r for r in config_results if r["variant"] == "Baseline"), None)
    if baseline and baseline["valid"]:
        best = min((r for r in config_results if r["valid"]), key=lambda x: x["timesteps"])
        improvement = (baseline["timesteps"] - best["timesteps"]) / baseline["timesteps"] * 100
        print(
            f"  Best: {best['variant']} with {best['timesteps']:.0f} steps "
            f"({improvement:+.1f}% vs baseline)"
        )
    print()

print("Benchmark complete!")
